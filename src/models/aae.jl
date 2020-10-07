struct AAE{P<:ContinuousMultivariateDistribution,E<:ConditionalMvNormal,D<:ConditionalMvNormal,F}
	prior::P
	encoder::E
	decoder::D
	discriminator::F
end

"""
	AAE(prior, encoder, decoder, discriminator)
	AAE(zdim::Int, encoder, decoder, discriminator)

Constructs an adversarial autoencoder.
"""
function AAE(zdim::Int, encoder, decoder, discriminator)
	W = first(Flux.params(encoder))
    μ = fill!(similar(W, zdim), 0)
    σ = fill!(similar(W, zdim), 1)
    prior = DistributionsAD.TuringMvNormal(μ, σ)
    AAE(prior, encoder, decoder, discriminator)
end

Flux.@functor AAE

function Flux.trainable(m::AAE)
    (encoder=m.encoder, decoder=m.decoder, discriminator=m.discriminator)
end
function Flux.trainable(m::AAE{<:VAMP})
    (prior=m.prior, encoder=m.encoder, decoder=m.decoder, discriminator=m.discriminator)
end

"""
	aae_constructor(;idim::Int=1, zdim::Int=1, activation = "relu", hdim=128, nlayers::Int=3, 
		init_seed=nothing, prior="normal", pseudoinput_mean=nothing, k=1, kwargs...)

Constructs an adversarial autoencoder with encoder-decoder-discriminator components.

# Arguments
- `idim::Int`: input dimension.
- `zdim::Int`: latent space dimension.
- `activation::String="relu"`: activation function.
- `hdim::Int=128`: size of hidden dimension.
- `nlayers::Int=3`: number of decoder/encoder layers, must be >= 3. 
- `init_seed=nothing`: seed to initialize weights.
- `prior="normal"`: one of ["normal", "vamp"].
- `pseudoinput_mean=nothing`: mean of data used to initialize the VAMP prior.
- `k::Int=1`: number of VAMP components. 
"""
function aae_constructor(;idim::Int=1, zdim::Int=1, activation = "relu", hdim=128, nlayers::Int=3, 
	init_seed=nothing, prior="normal", pseudoinput_mean=nothing, k=1, kwargs...)
	(nlayers < 2) ? error("Less than 3 layers are not supported") : nothing
	
	# if seed is given, set it
	(init_seed != nothing) ? Random.seed!(init_seed) : nothing
	
	# construct the model
	# encoder - diagonal covariance
	encoder_map = Chain(
		build_mlp(idim, hdim, hdim, nlayers-1, activation=activation)...,
		ConditionalDists.SplitLayer(hdim, [zdim, zdim], [identity, safe_softplus])
		)
	encoder = ConditionalMvNormal(encoder_map)
	
	# decoder - we will optimize only a shared scalar variance for all dimensions
	decoder_map = Chain(
		build_mlp(zdim, hdim, hdim, nlayers-1, activation=activation)...,
		ConditionalDists.SplitLayer(hdim, [idim, 1], [identity, safe_softplus])
		)
	decoder = ConditionalMvNormal(decoder_map)

	# discriminator
	discriminator = build_mlp(zdim, hdim, 1, nlayers, activation=activation, lastlayer="σ")

	# prior
	if prior == "normal"
		prior_arg = zdim
	elseif prior == "vamp"
		(pseudoinput_mean == nothing) ? error("if `prior=vamp`, supply pseudoinput array") : nothing
		prior_arg = init_vamp(pseudoinput_mean, k)
	end

	# reset seed
	(init_seed != nothing) ? Random.seed!() : nothing

	# get the vanilla VAE
	model = AAE(prior_arg, encoder, decoder, discriminator)
end

"""
    dloss(d,g,x,z)

Discriminator loss.
"""
dloss(d,g,x,z) = - 0.5f0*(mean(log.(d(x) .+ eps(Float32))) + mean(log.(1 .- d(g(z)) .+ eps(Float32))))

"""
    gloss(d,g,x)

Generator loss.
"""
gloss(d,g,x) = - mean(log.(d(g(x)) .+ eps(Float32)))

"""
	aeloss(AAE, x[, batchsize])

Autoencoding loss.
"""
aeloss(m::AAE,x) = - mean(logpdf(m.decoder, x, rand(m.encoder, x)))
aeloss(m::AAE,x,batchsize::Int) = 
	mean(map(y->aeloss(m,y), Flux.Data.DataLoader(x, batchsize=batchsize)))

"""
	dloss(AAE,x[, batchsize])

Discriminator loss given original sample x.
"""
dloss(m::AAE,x) = dloss(m.discriminator, y->mean(m.encoder,y), rand(m.prior, size(x,ndims(x))), x)
dloss(m::AAE{<:VAMP},x) = dloss(m.discriminator, y->mean(m.encoder,y), mean(m.encoder, rand(m.prior, size(x,ndims(x)))), x)
dloss(m::AAE,x,batchsize::Int) = 
	mean(map(y->dloss(m,y), Flux.Data.DataLoader(x, batchsize=batchsize)))
# note that X and Z is swapped here from the normal notation

"""
	gloss(AAE,x[, batchsize])

Encoder/generator loss.
"""
gloss(m::AAE,x) = gloss(m.discriminator, y->mean(m.encoder,y), x)
gloss(m::AAE,x,batchsize::Int) = 
	mean(map(y->gloss(m,y), Flux.Data.DataLoader(x, batchsize=batchsize)))

"""
	loss(m::AAE, x)

A loss that is used for stopping the training.
"""
loss(m::AAE,x) = aeloss(m,x)
loss(m::AAE,x,batchsize::Int) = aeloss(m,x,batchsize)
#loss(m::AAE,x) = aeloss(m,x) + dloss(m,x) + gloss(m,x)
#loss(m::AAE,x,batchsize::Int) = aeloss(m,x,batchsize) + dloss(m,x,batchsize) + gloss(m,x,batchsize)

"""
	StatsBase.fit!(model::AAE, data::Tuple; max_train_time=82800, 
	lr=0.001, batchsize=64, patience=30, check_interval::Int=10, kwargs...)
"""
function StatsBase.fit!(model::AAE, data::Tuple; max_train_time=82800, lr=0.001, 
	batchsize=64, patience=30, check_interval::Int=10, kwargs...)
	history = MVHistory()
	aeopt = ADAM(lr)
	dopt = ADAM(lr)
	gopt = ADAM(lr)

	tr_model = deepcopy(model)
	aeps = (typeof(tr_model.prior) <: GenerativeModels.VAMP) ? 
		Flux.params(tr_model.prior, tr_model.encoder, tr_model.decoder) : 
		Flux.params(tr_model.encoder, tr_model.decoder)
	dps = Flux.params(tr_model.discriminator)
	gps = Flux.params(tr_model.encoder)
	_patience = patience

	tr_x = data[1][1]
	val_x = data[2][1][:,data[2][2] .== 0]
	val_N = size(val_x,2)

	# on large datasets, batching loss is faster
	best_val_loss = Inf
	i = 1
	start_time = time() # end the training loop after 23hrs
	for batch in RandomBatches(tr_x, batchsize)
		# ae loss
		batch_aeloss = 0f0
		gs = gradient(() -> begin 
			batch_aeloss = aeloss(tr_model,batch)
		end, aeps)
	 	Flux.update!(aeopt, aeps, gs)

	 	# disc loss
		batch_dloss = 0f0
		gs = gradient(() -> begin 
			batch_dloss = dloss(tr_model,batch)
		end, dps)
	 	Flux.update!(dopt, dps, gs)

	 	# gen loss
		batch_gloss = 0f0
		gs = gradient(() -> begin 
			batch_gloss = gloss(tr_model,batch)
		end, gps)
	 	Flux.update!(gopt, gps, gs)

		# validation
		val_loss = (val_N > 5000) ? loss(tr_model, val_x, 256) : loss(tr_model, val_x)
		(i%check_interval == 0) ? (@info "$i - loss: $(batch_aeloss) (autoencoder) $(batch_dloss) (discriminator) $(batch_gloss) (generator)  | $(val_loss) (validation)") : nothing
		
		# check nans
		if isnan(val_loss) || isnan(batch_aeloss) || isnan(batch_dloss) || isnan(batch_gloss)
			error("Encountered invalid values in loss function.")
		end

		# save training progress
		push!(history, :training_aeloss, i, batch_aeloss)
		push!(history, :training_dloss, i, batch_dloss)
		push!(history, :training_gloss, i, batch_gloss)
		push!(history, :validation_loss, i, val_loss)
			
		# early stopping
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
			@info "Stopped training after $(i) iterations, $((time() - start_time)/3600) hours."
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
	# again, this is not optimal, the model should be passed by reference and only the reference should be edited
	(history=history, iterations=i, model=model, npars=sum(map(p->length(p), Flux.params(model))))
end

"""
	AEModel

A Union of VAE and AAE types.
"""
AEModel = Union{VAE, AAE}

"""
	reconstruct(model::AEModel, x)

Data reconstruction.
"""
reconstruct(model::AEModel, x) = mean(model.decoder, rand(model.encoder, x))

"""
	reconstruction_score(model::AEModel, x)

Anomaly score based on the reconstruction probability of the data.
"""
function reconstruction_score(model::AEModel, x) 
	p = condition(model.decoder, rand(model.encoder, x))
	-logpdf(p, x)
end
"""
	reconstruction_score_mean(model::AEModel, x)

Anomaly score based on the reconstruction probability of the data. Uses mean of encoding.
"""
function reconstruction_score_mean(model::AEModel, x) 
	p = condition(model.decoder, mean(model.encoder, x))
	-logpdf(p, x)
end
"""
	latent_score(model::AEModel, x) 

Anomaly score based on the similarity of the encoded data and the prior.
"""
function latent_score(model::AEModel, x) 
	z = rand(model.encoder, x)
	-logpdf(model.prior, z)
end

"""
	latent_score_mean(model::AEModel, x) 

Anomaly score based on the similarity of the encoded data and the prior. Uses mean of encoding.
"""
function latent_score_mean(model::AEModel, x) 
	z = mean(model.encoder, x)
	-logpdf(model.prior, z)
end
