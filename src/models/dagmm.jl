using Flux: mse, softmax, unsqueeze, stack
using LinearAlgebra
using StatsBase
using Statistics
using Distributions
using DistributionsAD

struct DAGMM{E,D,S}
    encoder::E
    decoder::D
    estimator::S
end

Flux.@functor DAGMM

function DAGMM(;idim::Int=1, activation="tanh", hdim=60, zdim=1, ncomp=2, 
        nlayers::Int=3, dropout=0.5, init_seed=nothing, kwargs...)
    # if seed is given, set it
    (init_seed !== nothing) ? Random.seed!(init_seed) : nothing

    encoder = build_mlp(idim, hdim, zdim, nlayers, activation=activation, lastlayer="linear")
    decoder = build_mlp(zdim, hdim, idim, nlayers, activation=activation, lastlayer="linear")
    estimator = build_mlp(zdim+2, hdim, ncomp, nlayers, activation=activation, lastlayer="linear")
    
    if dropout > 0.0
        estimator = Chain(estimator[1:end-1]..., Dropout(dropout), estimator[end])
    end

    # reset seed
    (init_seed !== nothing) ? Random.seed!() : nothing

    DAGMM(encoder, decoder, estimator)
end

function Base.show(io::IO, model::DAGMM)
    # to avoid the show explosion
    print(io, "DAGMM(...)")
end

norm2(x; dims=1) = sqrt.(sum(abs2, x; dims=dims))

function cosine_similarity(x₁, x₂; dims=1, eps=1e-8)
    T = eltype(x₁)
    nx₁ = norm2(x₁; dims=dims)
    nx₂ = norm2(x₂; dims=dims)
    sum(x₁ .* x₂; dims=dims) ./ max.(nx₁ .* nx₂, T(eps))
end

function compute_reconstruction(x, x̂)
    relrec = norm2(x .- x̂; dims=1) ./ norm2(x; dims=1)
    cosim = cosine_similarity(x, x̂; dims=1)
    relrec, cosim
end

function (model::DAGMM)(x)
    z_c = model.encoder(x)
    x̂ = model.decoder(z_c)
    rec_1, rec_2 = compute_reconstruction(x, x̂)
    z = cat([z_c, rec_1, rec_2]..., dims=1)
    gamma = softmax(model.estimator(z))
    z_c, x̂, z, gamma
end


"""
Computing the parameters phi, mu and gamma for sample energy function 

K: number of Gaussian mixture components
N: Number of samples
D: Latent dimension

z = DxN
gamma = KxN

this will be a little bit different in Julia
""" 
function compute_params(z, gamma)
    #phi = D
    gamma_sum = sum(gamma; dims=2)
    phi = gamma_sum ./ size(gamma, 2)

    #mu = D x K x 1
    mu = sum(unsqueeze(gamma,1) .* unsqueeze(z,2); dims=3) ./ gamma_sum'

    # z_mu = D x K x N
    z_mu = unsqueeze(z,2) .- mu
    
    # z_mu = D x D x K x N
    z_mu_z_mu_t = unsqueeze(z_mu, 2) .* unsqueeze(z_mu, 1)
    
    #cov = K x D x D
    cov = sum(unsqueeze(unsqueeze(gamma, 1), 1) .* z_mu_z_mu_t; dims=4)
    cov = dropdims(cov; dims=4) ./ unsqueeze(gamma_sum', 1)

    phi, mu, cov
end

function compute_energy(z, phi, mu, cov; eps=1e-12)
    T = eltype(z)
    z_mu = unsqueeze(z,2) .- mu
    D, K, _ = size(mu)

    cov = cov .+ Diagonal(ones(Float32, D) .* T(eps))
  
    cov_inverse = stack([inv(cov[:,:,k]) for k in 1:K], 3)
    det_cov = unsqueeze(unsqueeze([det(cov[:,:,k] .* 2 * T.(π)) for k in 1:K],1),1)
    cov_diag = sum([sum(1 ./ diag(cov[:,:,k])) for k in 1:K])

    E_z = -T(0.5) .* sum(sum(unsqueeze(z_mu, 1) .* cov_inverse, dims=2) .* unsqueeze(z_mu, 2), dims=1)
    E_z = exp.(E_z)
    E_z = -log.(sum(unsqueeze(phi', 1).* E_z ./ sqrt.(det_cov), dims=3).+ T(eps))
          
    E_z, cov_diag
end


function loss(model::DAGMM, x, λ₁, λ₂)
    _, x̂, z, gamma = model(x)
    reconst_loss = mse(x, x̂)
    phi, mu, cov = compute_params(z, gamma)
    sample_energy, cov_diag = compute_energy(z, phi, mu, cov)
    reconst_loss + λ₁ * mean(sample_energy) + λ₂ * cov_diag
end

# during testing we use params fitted from training
# testmode is not really necessary because dropout is not part if the autoencoder
function StatsBase.predict(model::DAGMM, x, phi, mu, cov)
    testmode!(model, true)
    _, _, z, _ = model(x)
    testmode!(model, false)
    compute_energy(z, phi, mu, cov)[1]
end


function StatsBase.fit!(model::DAGMM, data::Tuple; max_train_time=82800,
                        batchsize=64, patience=200, check_interval::Int=1, 
                        wreg=1f-6, lambda_rat=1, lr=1f-4, kwargs...)
    # add regularization through weight decay in optimizer
    opt = (wreg > 0) ? ADAMW(lr, (0.9, 0.999), wreg) : Flux.ADAM(lr)
    
    trn_model = deepcopy(model)
    ps = Flux.params(trn_model);

    X = data[1][1]
    # filter only normal data from validation set
    X_val = data[2][1][:, data[2][2] .== 0.0f0]

    # hardcoded parameters, changing only the ratios
    λ₁, λ₂ = 0.1f0 * lambda_rat, 0.005f0 * lambda_rat

    history = MVHistory()
    _patience = patience
    
    best_val_loss = Inf
    i = 1
    start_time = time()
    frmt(v) = round(v, digits=4)
    for batch in RandomBatches(X, batchsize)
        batch_loss = 0f0

        grad_time = @elapsed begin
            gs = gradient(() -> begin 
                batch_loss = loss(trn_model, batch, λ₁, λ₂)
            end, ps)
            Flux.update!(opt, ps, gs)
        end

        push!(history, :batch_loss, i, batch_loss)
        push!(history, :grad_time, i, grad_time)

        if (i%check_interval == 0)
            testmode!(trn_model, true)
            val_loss_time = @elapsed val_loss = loss(trn_model, X_val, λ₁, λ₂)
            testmode!(trn_model, false)

            @info "$i - loss: $(frmt(batch_loss)) (batch) | $(frmt(val_loss)) (validation) || $(frmt(grad_time)) (t_grad) | $(frmt(val_loss_time)) (t_val)"
            
            if isnan(val_loss) || isinf(val_loss) || isnan(batch_loss) || isinf(batch_loss)
                @info "Stopped training after $(i) iterations, $((time() - start_time)/3600) hours due to invalid values."
                error("Encountered invalid values in loss function.")
            end

            push!(history, :validation_loss, i, val_loss)
            push!(history, :val_loss_time, i, val_loss_time)

            if val_loss < best_val_loss
                best_val_loss = val_loss
                _patience = patience

                model = deepcopy(trn_model)
            else # else stop if the model has not improved for `patience` iterations
                _patience -= 1
                if _patience == 0
                    @info "Stopped training after $(i) iterations, $((time() - start_time)/3600) hours."
                    break
                end
            end
        end
        
        # time limit for training
        if time() - start_time > max_train_time
            @info "Stopped training after $(i) iterations, $((time() - start_time)/3600) hours due to time constraints."
            model = deepcopy(trn_model)
            break
        end

        i += 1
    end

    (history=history, niter=i, model=model, npars=sum(map(p->length(p), ps)))
end
