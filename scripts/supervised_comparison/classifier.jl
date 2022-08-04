using DrWatson
@quickactivate
using GenerativeAD
using Flux, StatsBase
using Random
using GenerativeAD.Models: conv_encoder
using EvalMetrics, IterTools, ValueHistories

function sample_params()
	# first sample the number of layers
	nlayers = rand(2:4)
	kernelsizes = reverse((3,5,7,9)[1:nlayers])
	channels = reverse((16,32,64,128)[1:nlayers])
	scalings = reverse((1,2,2,2)[1:nlayers])
	
	par_vec = (10f0 .^(-4:-3), 2 .^ (5:7), ["relu", "swish"], 1:Int(1e8), [true, false])
	argnames = (:lr, :batchsize, :activation, :init_seed, :batchnorm)
	parameters = (;zip(argnames, map(x->sample(x, 1)[1], par_vec))...)
	return merge(parameters, (nlayers=nlayers, kernelsizes=kernelsizes,
		channels=channels, scalings=scalings))
end

function GenerativeAD.edit_params(x, parameters)
	idim = size(x)
	# on MNIST and FashionMNIST 4 layers are too much
	if parameters.nlayers >= 4 && idim[1]*idim[2] >= 28*28
		nlayers = rand(2:3)
		kernelsizes = reverse((3,5,7,9)[1:nlayers])
		channels = reverse((16,32,64,128)[1:nlayers])
		scalings = reverse((1,2,2,2)[1:nlayers])
		parameters = merge(parameters, (nlayers=nlayers,kernelsizes=kernelsizes,channels=channels,scalings=scalings))
	end
	parameters
end

struct Classifier
    map
    lr
    batchsize
end

(m::Classifier)(x) = m.map(x)

Flux.@functor Classifier

function classifier_constructor(;idim=(32,32,3), batchsize = 32, activation = "relu", init_seed = nothing,
        nlayers = 3, kernelsizes = (7, 5, 3), channels = (64, 32, 16), scalings = (2, 2, 1), batchnorm=false,
        lr = 0.001f0)

    # if seed is given, set it
    (init_seed != nothing) ? Random.seed!(init_seed) : nothing

    # first get the main layers
    map = conv_encoder(idim, 2, kernelsizes, channels, scalings; activation=activation,
        batchnorm=batchnorm)
    
    # reset seed
    (init_seed !== nothing) ? Random.seed!() : nothing

    return Classifier(map, lr, batchsize)
end

function fit_classifier(tr_x, tr_y, tst_x, tst_y, parameters)
	# cosntruct the model
	model = classifier_constructor(;parameters...) |> gpu

	# minibatch loader - equal samples from both classes
	labelnames = unique(val_y)
	function minibatch()
	    nb = Int(model.batchsize/2)
	    n1 = Int(sum(val_y))
	    n0 = length(val_y) - n1
	    idx1 = sample(1:n1, nb, replace=false)
	    idx0 = sample(1:n0, nb, replace=false)
	    T = eltype(val_y)
	    inds = Bool.(val_y)
	    x = gpu(cat(val_x[:,:,:, .!inds][:,:,:,idx0], val_x[:,:,:,inds][:,:,:,idx1], dims=4))
	    y = gpu(Flux.onehotbatch(vcat(zeros(T, nb), ones(T,nb)), labelnames))
	    x, y
	end

	# better metrics
	probs(x) = batch_eval(y->model(y), x)
	predict(ps::AbstractArray{T,2}) where T = labelnames[Flux.onecold(ps)]
	predict(x) = predict(probs(x))
	accuracy(y_true, y_pred) = mean(y_true .== y_pred)
	accuracy(y_true, y_pred, class) = (inds = y_true .== class; accuracy(y_true[inds], y_pred[inds]))
	precision(y_true, y_pred) = (tp = sum(y_pred[y_true .== 1]); fp = sum(y_pred[y_true .== 0]); tp/(tp+fp))
	predict_scores(ps) = ps[findfirst(labelnames .== 1),:]
	auc_val(labels, scores) = auc_trapezoidal(roccurve(labels, scores)...)

	# callback and history
	history = MVHistory()
	i = 0
	cb = () -> begin
		# get scores
	    testmode!(model)
		ps = map(probs, (val_x, tst_x))
		y_preds = map(predict, ps)
		y_trues = (val_y, tst_y)
	    trainmode!(model)

		# precisions	
		tr_acc, tst_acc = map(y->round(accuracy(y[1], y[2]),digits=3), zip(y_trues, y_preds))
		tr_prec, tst_prec = map(y->round(precision(y[1], y[2]),digits=3), zip(y_trues, y_preds))
		tr_acc_pos, tst_acc_pos = map(y->round(accuracy(y[1], y[2], 1),digits=3), zip(y_trues, y_preds))
		tr_acc_neg, tst_acc_neg = map(y->round(accuracy(y[1], y[2], 0),digits=3), zip(y_trues, y_preds))
		tr_auc, tst_auc = map(y->round(auc_val(y[1], predict_scores(y[2])), digits=3), zip(y_trues, ps))

		println("                 acc |prec |TPR  |FPR  |AUC: 
	        train = $(tr_acc)|$(tr_prec)|$(tr_acc_pos)|$(tr_acc_neg)|$(tr_auc), 
	        test  = $(tst_acc)|$(tst_prec)|$(tst_acc_pos)|$(tst_acc_neg)|$(tst_auc)")
	    
	    # save to history
	    i += 1
	    push!(history, :tr_acc, i, tr_acc)
	    push!(history, :tst_acc, i, tst_acc)
	    push!(history, :tr_prec, i, tr_prec) 
	    push!(history, :tst_prec, i, tst_prec)
	    push!(history, :tr_acc_pos, i, tr_acc_pos) 
	    push!(history, :tst_acc_pos, i, tst_acc_pos)
	    push!(history, :tr_acc_neg, i, tr_acc_neg) 
	    push!(history, :tst_acc_neg, i, tst_acc_neg)
	    push!(history, :tr_auc, i, tr_auc) 
	    push!(history, :tst_auc, i, tst_auc)
	end

	# rest of the setup
	iterations = 2000
	ps = Flux.params(model);
	loss = (x,y) -> Flux.logitcrossentropy(model(x), y)
	opt = ADAM(model.lr)

	# train
	Flux.Optimise.train!(loss, ps, repeatedly(minibatch, iterations), opt, cb = Flux.throttle(cb, 2))

	# compute the interesting values - tst and val auc
    testmode!(model)
	ps = map(probs, (val_x, tst_x))
	y_trues = (val_y, tst_y)
    trainmode!(model)

	# precisions	
	tr_auc, tst_auc = map(y->auc_val(y[1], predict_scores(y[2])), zip(y_trues, ps))

	# return the predicted values
	return model, history, tr_auc, tst_auc
end

batch_eval(scoref, x, batchsize=512) =
    hcat(map(y->cpu(scoref(gpu(Array(y)))), Flux.Data.DataLoader(x, batchsize=batchsize))...)