"""
	Implements all needed parts for constructing DeepSVDD model for detecting anomalies.
		(Depp One-Class Classification -> http://proceedings.mlr.press/v80/ruff18a/ruff18a.pdf) 
	Code is inspired by original Pytorch implementation https://github.com/lukasruff/Deep-SVDD-PyTorch.

"""

mutable struct DeepSVDD
	encoder
	decoder
	objective
	c
	R
	nu
end

Flux.@functor DeepSVDD

function (svdd::DeepSVDD)(x)
	z = svdd.encoder(x)
	return z
end

function AE(svdd::DeepSVDD, x)
	z = svdd.encoder(x)
	x̂ = svdd.decoder(z)
	return x̂
end

function anomaly_score(svdd::DeepSVDD, x)
	z = svdd(x)
	dist = vec(Flux.sum((z .- svdd.c).^2, dims=1))
	score = (svdd.objective == "soft-boundary") ? dist .- svdd.R.^2 : dist
	return score
end

function init_c(svdd::DeepSVDD, data, eps=0.1; batchsize=50, to_testmode::Bool=true)
	(to_testmode == true) ? Flux.testmode!(svdd) : nothing
	c_ = deepcopy(svdd.c)
	for X in Flux.Data.DataLoader(data, batchsize=batchsize)
		X = getobs(X)|>gpu
		ŷ = svdd(X)
		c_ += Flux.sum(ŷ, dims=2) 
	end
	c_ = c_ |> cpu
	c_ /= size(data, 4)
	c_[(abs.(c_) .< eps) .& (c_ .< 0)] .= -eps
	c_[(abs.(c_) .< eps) .& (c_ .> 0)] .= eps
	(to_testmode == true) ? Flux.testmode!(svdd, false) : nothing
	return c_ 
end

function get_radius(svdd::DeepSVDD, X)
	output = svdd(X)
	dist = vec(Flux.sum((output .- svdd.c).^2, dims=1))
	return Statistics.quantile(sqrt.(dist), 1-svdd.nu)
end

# LOSSES

# autoencoder loss function
ae_loss(ŷ, y) = Flux.mean(Flux.sum((ŷ .- y).^2, dims=(1,2,3))) 

# objective / loss function for Soft-boundary "SVM"
function sb_loss(ŷ, c, R, nu)
	dist = Flux.sum((ŷ .- c).^2, dims=1)
	scores = dist .- R^2 # 
	loss = R^2 + (1 / nu) * Flux.mean(max.(0, scores))
end

# objective / loss function for One-Class "SVM"
oc_loss(ŷ, c, R, nu) = Flux.mean(Flux.sum((ŷ .- c).^2, dims=1))

# anomaly score on GPU
function anomaly_score_gpu(svdd::DeepSVDD, real_input; batch_size=64, to_testmode::Bool=true)
	real_input = Flux.Data.DataLoader(real_input, batchsize=batch_size)
	(to_testmode == true) ? Flux.testmode!(svdd) : nothing
	svdd = svdd |> gpu
	scores = Array{Float32}([])
	for x in real_input
		x = x |> gpu
		score = anomaly_score(svdd, x)
		scores = cat(scores, score |> cpu, dims=1)
	end
	(to_testmode == true) ? Flux.testmode!(svdd, false) : nothing
	return scores
end

######################################################################################

function conv_ae_constructor(
	;idim=(2,2,1), 
	zdim::Int=1, 
	activation="relu", 
	kernelsizes=(1,1), 
	channels=(1,1), 
	scalings=(1,1),
	init_seed=nothing, 
	objective="soft-boundary",
	batchnorm=false, 
	nu::Float32 = 0.1f0,
	R::Float32 = 0f0,
	c = nothing,
	kwargs...
)
	# if seed is given, set it
	(init_seed !== nothing) ? Random.seed!(init_seed) : nothing
	# encoder 
	encoder = Chain(
		conv_encoder(idim, zdim, kernelsizes, channels, scalings; activation=activation, batchnorm=batchnorm)...
	)
	# decoder 
	decoder = Chain(
		conv_decoder(idim, zdim, reverse(kernelsizes), reverse(channels), reverse(scalings),
			activation=activation, vec_output=false, vec_output_dim=nothing, batchnorm=batchnorm)...,
			x->sigmoid.(x) # our dataset images are scaled from 0-1
	)
	# reset seed
	(init_seed !== nothing) ? Random.seed!() : nothing

	c_ = (c !== nothing) ? c : zeros(Float32, zdim)

	model = DeepSVDD(encoder, decoder, objective, c_, R, nu)
end

##################################################################################

# fit Autoencoder
function fit_ae(svdd::DeepSVDD, optim, data, params)
	train_loader, val_loader = prepare_dataloaders(data, batch_size=params.batch_size, iters=params.ae_iters)
	history = MVHistory()
	total_iters = length(train_loader)
	progress = Progress(total_iters)
	# early stopping
	best_svdd = deepcopy(svdd)
	patience = params.patience
	val_batches = length(val_loader)
	best_val_loss = 1e10

	svdd = svdd |> gpu
	ps_ae = Flux.params(svdd.encoder, svdd.decoder)
	for (iter, X) in enumerate(train_loader)
		X = getobs(X)|>gpu
		loss_ae, back = Flux.pullback(ps_ae) do
			ae_loss(AE(svdd, X), X)
		end
		grad_ae = back(1f0)
		Flux.Optimise.update!(optim, ps_ae, grad_ae)
		
		push!(history, :loss_ae, iter, loss_ae)

		next!(progress; showvalues=[
			(:iters, "$(iter)/$(total_iters)"),
			(:ae_loss, loss_ae)
			])

		if mod(iter, params.check_every) == 0
			total_val_loss = 0
			Flux.testmode!(svdd)
			for X_val in val_loader
				X_val = X_val |> gpu
				total_val_loss += ae_loss(AE(svdd, X), X)
			end
			Flux.testmode!(svdd, false)
			push!(history, :validation_ae_loss, iter, total_val_loss/val_batches)
			if total_val_loss < best_val_loss
				best_val_loss = total_val_loss
				patience = params.patience
				best_svdd = deepcopy(svdd)
			else
				patience -= 1
				if patience == 0
					@info "Stopped training of Autoencoder after $(iter) iterations"
					break
				end
			end
		end        
	end
	return best_svdd, history
end


function StatsBase.fit!(svdd::DeepSVDD, data, params)
	# prepare batches & loaders
	train_loader, val_loader = prepare_dataloaders(data, params)
	svdd_history = MVHistory()
	# prepare for early stopping
	best_val_loss = 1e10
	
	patience = params.patience
	val_batches = length(val_loader)

	#model to gpu
	svdd = svdd |> gpu

	# ADAMW(η = 0.001, β = (0.9, 0.999), decay = 0) = Optimiser(ADAM(η, β), WeightDecay(decay))
	opt_ae = haskey(params, :decay) ? ADAMW(params.lr_ae, (0.9, 0.999), params.decay) : ADAM(params.lr_ae)
	opt_svdd = haskey(params, :decay) ? ADAMW(params.lr_svdd, (0.9, 0.999), params.decay) : ADAM(params.lr_svdd)

	# loss_function for SVDD
	loss = (svdd.objective == "soft-boundary") ? sb_loss : oc_loss

	# Pretraining of AE
	svdd, ae_history = fit_ae(svdd, opt_ae, data, params)
	best_svdd = deepcopy(svdd)

	# init center of hypershpere
	svdd.c = init_c(svdd, data[1][1]) |> gpu

	# train "SVDD"
	progress = Progress(params.iters)
	ps_svdd = Flux.params(svdd.encoder)
	for (iter, X) in enumerate(train_loader)
		X = getobs(X)|>gpu
		loss_svdd, back = Flux.pullback(ps_svdd) do
			loss(svdd(X), svdd.c, svdd.R, svdd.nu)
		end
		grad_svdd = back(1f0)
		Flux.Optimise.update!(opt_svdd, ps_svdd, grad_svdd)

		# after warm up 
		svdd.R = ((iter >= 0.05*params.iters) & (svdd.objective == "soft-boundary")) ? get_radius(svdd, X) : svdd.R

		push!(svdd_history, :loss_svdd, iter, loss_svdd)
		next!(progress; showvalues=[
			(:iters, "$(iter)/$(params.iters)"),
			(:svdd_loss, loss_svdd)
			])

		if mod(iter, params.check_every) == 0
			total_val_loss = 0
			Flux.testmode!(svdd)
			for X_val in val_loader
				X_val = X_val |> gpu
				total_val_loss += loss(svdd(X), svdd.c, svdd.R, svdd.nu)
			end
			Flux.testmode!(svdd, false)
			push!(svdd_history, :validation_svdd_loss, iter, total_val_loss/val_batches)
			if total_val_loss < best_val_loss
				best_val_loss = total_val_loss
				patience = params.patience
				best_svdd = deepcopy(svdd)
			else
				patience -= 1
				if patience == 0
					@info "Stopped training after $(iter) iterations"
					global iters = iter - params.check_every*params.patience
					break
				end
			end
		end
		if iter == params.iters
			global iters = params.iters
		end
	end

	return (svdd_history, ae_history), best_svdd, sum(map(p->length(p), Flux.params(svdd))), iters
end
