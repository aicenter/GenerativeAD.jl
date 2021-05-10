"""
    Implements all needed parts for constructing REPEN model
        (http://arxiv.org/abs/1806.04808) for detecting anomalies.
    Code is inspired by original Keras implementation 
    https://github.com/GuansongPang/deep-outlier-detection
    https://github.com/Ryosaeba8/Anomaly_detection/tree/master/REPEN
"""

using Random
using Statistics
using StatsBase: sample, predict, fit!, Weights
using NearestNeighbors: KDTree, knn, Euclidean
using Flux

struct REPEN{E}
    e::E
end

Flux.@functor REPEN

function REPEN(;idim::Int=1, activation="relu", hdim=40, 
                zdim=20, nlayers::Int=1, init_seed=nothing, kwargs...)
    # if seed is given, set it
    (init_seed !== nothing) ? Random.seed!(init_seed) : nothing

    e = build_mlp(idim, hdim, zdim, nlayers, activation=activation)
    
    # reset seed
    (init_seed !== nothing) ? Random.seed!() : nothing

    REPEN(e)
end    

function (m::REPEN)(x::Tuple)
    @assert length(x) == 3
    examples, positive, negative = x
    m.e(examples), m.e(positive), m.e(negative)
end

function Base.show(io::IO, model::REPEN)
    print(io, "REPEN(")
    print(io, model.e)
    print(io, ")")
end

sqr_euclidean_dist(x,y; dims=1) = sum(abs2, x .- y, dims=dims)

function ranking_loss(input_example, input_positive, input_negative;
                        conf_margin=1000.0)
    T = eltype(input_example)
    positive_distances = sqr_euclidean_dist(input_example, input_positive)
    negative_distances = sqr_euclidean_dist(input_example, input_negative)
    
    mean(max.(T(0.0), T(conf_margin) .- (negative_distances .- positive_distances)))
end

function lesinn(xq; fit_data=xq, ensemble_size=50, subsample_size=8, kwargs...)
    Random.seed!(42) 
    n = size(fit_data, 2)
    scores = zeros(Float32, size(xq, 2))
    for i in 1:ensemble_size
        subsample = fit_data[:, randperm(n)[1:min(subsample_size, round(Int, 3n//4))]]
        kdt = KDTree(subsample, Euclidean(); leafsize = 40)
        _, dists = knn(kdt, xq, 1)
        scores .+= getindex.(dists, 1)
    end
    Random.seed!() 
    scores ./ ensemble_size
end

function cutoff_unsorted(vals; th0=1.7321)
    v_mean = mean(vals)
    v_std = std(vals)
    th = v_mean + th0 * v_std 
    if th >= maximum(vals) # return the top-10 outlier scores
        temp = sort(vals)
        th = temp[end-10]
    end

    outlier_ind = findall(vals .> th)
    inlier_ind = findall(vals .<= th)
    
    inlier_ind, outlier_ind
end


function triplet_batch_generation(X, outlier_scores; batchsize=256)
    inlier_ids, outlier_ids = cutoff_unsorted(outlier_scores)
    nin, nout = length(inlier_ids), length(outlier_ids)
    
    transforms = sum(outlier_scores[inlier_ids]) .- outlier_scores[inlier_ids]
    total_weights_p = sum(transforms)
    positive_weights = transforms ./ total_weights_p
    
    total_weights_n = sum(outlier_scores[outlier_ids])
    negative_weights = outlier_scores[outlier_ids] ./ total_weights_n
    
    examples_ids = zeros(Int, batchsize)
    positives_ids = zeros(Int, batchsize)
    negatives_ids = zeros(Int, batchsize)
    
    for i in 1:batchsize
        sid = sample(1:nin, Weights(positive_weights))
        examples_ids[i] = inlier_ids[sid]
        
        sid2 = sample(1:nin)
        
        while sid2 == sid
            sid2 = sample(1:nin)
        end
        positives_ids[i] = inlier_ids[sid2]

        sid = sample(1:nin, Weights(negative_weights))
        negatives_ids[i] = outlier_ids[sid]
    end
    
    X[:, examples_ids], X[:, positives_ids], X[:, negatives_ids]
end


"""
    function StatsBase.predict(model::REPEN, xq; fit_data=xq, ensemble_size=50, subsample_size=8)

Computes LESINN score by constructing KDTree over subsamples of *projected* 
training data `fit_data` and predicts on *projected* query points `xq` using 1NN.
"""
function StatsBase.predict(model::REPEN, xq; fit_data=xq, ensemble_size=50, subsample_size=8, kwargs...)
    zq = model.e(xq)
    zfit = model.e(fit_data)

    lesinn(zq; fit_data=zfit, ensemble_size=ensemble_size, subsample_size=subsample_size)
end


function StatsBase.fit!(model::REPEN, data::Tuple; max_train_time=82800,
                        batchsize=64, max_iters=10000, check_interval::Int=10, 
                        conf_margin=1000.0, kwargs...)
    
    opt = ADADelta()
    ps = Flux.params(model);

    X = data[1][1]
    outlier_scores = lesinn(X; kwargs...)
    history = MVHistory()

    start_time = time()
    frmt(v) = round(v, digits=4)
    for i in 1:max_iters
        batch_loss = 0f0
        sample_time = @elapsed batch = triplet_batch_generation(X, outlier_scores; batchsize=batchsize)

        grad_time = @elapsed begin
            gs = gradient(ps) do
                z_triplet = model(batch)
                batch_loss = ranking_loss(z_triplet...; conf_margin=conf_margin)
            end
            Flux.update!(opt, ps, gs)
        end

        push!(history, :batch_loss, i, batch_loss)
        push!(history, :grad_time, i, grad_time)

        if isnan(batch_loss) || isinf(batch_loss)
            @info "Stopped training after $(i) iterations, $((time() - start_time)/3600) hours due to invalid values."
            error("Encountered invalid values in loss function.")
        end

        if (i%check_interval == 0)
            @info "$i - loss: $(frmt(batch_loss)) (batch) | $(frmt(sample_time)) (t_batchgen) | $(frmt(grad_time)) (t_grad)"
        end
        
        # time limit for training
        if time() - start_time > max_train_time
            @info "Stopped training after $(i) iterations, $((time() - start_time)/3600) hours due to time constraints."
            model = deepcopy(model)
            break
        end

    end

    (history=history, niter=max_iters, model=model, npars=sum(map(p->length(p), ps)))
end