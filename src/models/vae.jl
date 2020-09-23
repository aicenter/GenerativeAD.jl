using Flux
using ConditionalDists
using GenerativeModels
using ValueHistories
using MLDataPattern: RandomBatches
using Distributions
using StatsBase

"""
	safe_softplus(x::T)

Safe version of softplus.	
"""
safe_softplus(x::T) where T  = softplus(x) + T(0.000001)


"""
	vae_constructor(;idim::Int=1, zdim::Int=1, activation = "relu", hdim=128, nlayers=3, kwargs...)
"""
function vae_constructor(;idim::Int=1, zdim::Int=1, activation = "relu", hdim=128, nlayers=3, kwargs...)
	(nlayers < 2) ? error("Less than 3 layers are not supported") : nothing
	# function from string
	act = eval(Meta.parse(activation))
	
	# construct the model
	# encoder - diagonal covariance
	encoder_map = Chain(
		Dense(idim,hdim,act),
		[Dense(hdim, hdim, act) for _ in 1:nlayers-2]...,
		ConditionalDists.SplitLayer(hdim, [zdim, zdim], [identity, safe_softplus])
		)
	encoder = ConditionalMvNormal(encoder_map)
	# decoder - we will optimize only a shared scalar variance for all dimensions
	decoder_map = Chain(
		Dense(zdim,hdim,act),
		Dense(hdim,hdim,act),
		ConditionalDists.SplitLayer(hdim, [idim, 1], [identity, safe_softplus])
		)
	decoder = ConditionalMvNormal(decoder_map)

	# get the vanilla VAE
	model = VAE(zdim, encoder, decoder)
end

"""
	loss(model::GenerativeModels.VAE, x[, batchsize])

Negative ELBO for training of a VAE model.
"""
loss(model::GenerativeModels.VAE, x) = -elbo(model, x)
# version of loss for large datasets where
loss(model::GenerativeModels.VAE, x, batchsize::Int) = 
	mean(map(y->loss(model,y), Flux.Data.DataLoader(x, batchsize=batchsize)))

"""
	StatsBase.fit!(model::GenerativeModels.VAE, data::Tuple; max_train_time=82800, lr=0.001, 
		batchsize=64, patience=30, check_interval::Int=10, kwargs...)
"""
function StatsBase.fit!(model::GenerativeModels.VAE, data::Tuple; max_train_time=82800, lr=0.001, 
	batchsize=64, patience=30, check_interval::Int=10, kwargs...)
	history = MVHistory()
	opt = ADAM(lr)

	tr_model = deepcopy(model)
	ps = Flux.params(tr_model)
	_patience = patience

	tr_x = data[1][1]
	val_x = data[2][1]
	val_N = size(val_x,2)

	# on large datasets, batching loss is faster
	best_val_loss = (val_N > 5000) ? loss(tr_model, val_x, 256) : loss(tr_model, val_x)
	i = 1
	start_time = time() # end the training loop after 23hrs
	for batch in RandomBatches(tr_x, batchsize)
		batch_loss = 0f0
		gs = gradient(() -> begin 
			batch_loss = loss(tr_model,batch)
		end, ps)
	 	Flux.update!(opt, ps, gs)

		# validation/early stopping
		val_loss = (val_N > 5000) ? loss(tr_model, val_x, 256) : loss(tr_model, val_x)
		
		(i%check_interval == 0) ? (@info "$i - loss: $(batch_loss) (batch) | $(val_loss) (validation)") : nothing
			
		if isnan(val_loss) || isnan(batch_loss)
			error("Encountered invalid values in loss function.")
		end

		push!(history, :training_loss, i, batch_loss)
		push!(history, :validation_likelihood, i, val_loss)
			
		if val_loss < best_val_loss
			best_val_loss = val_loss
			_patience = patience

			# this should save the model at least once
			# when the validation loss is decreasing 
			if mod(i, 10) == 0
				model = deepcopy(tr_model)
			end
		elseif time() - start_time > max_train_time # stop early if time is running out
			model = deepcopy(tr_model)
			@info "Stopped training after $((time() - start_time)/3600) hours."
			break
		else # else stop if the model has not improved for `patience` iterations
			_patience -= 1
			if _patience == 0
				@info "Stopped training after $(i) iterations."
				break
			end
		end
		i += 1
	end
	(history=history, iterations=i, model=model)
end

"""
	reconstruct(model::GenerativeModels.VAE, x)

Data reconstruction.
"""
reconstruct(model::GenerativeModels.VAE, x) = mean(model.decoder, rand(model.encoder, x))

"""
	reconstruction_score(model::GenerativeModels.VAE, x)

Anomaly score based on 
"""
function reconstruction_score(model::GenerativeModels.VAE, x) 
	p = condition(model.decoder, rand(model.encoder, x))
	-logpdf(p, x)
end
"""
	latent_score(model::GenerativeModels.VAE, x) 
"""
function latent_score(model::GenerativeModels.VAE, x) 
	z = rand(model.encoder, x)
	-logpdf(model.prior, z)
end
