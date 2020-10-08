using Flux
using ConditionalDists
using GenerativeModels
import GenerativeModels: GAN
using ValueHistories
using MLDataPattern: RandomBatches
using Distributions
using DistributionsAD
using StatsBase
using Random

"""
	gan_constructor(;idim::Int=1, zdim::Int=1, activation = "relu", hdim=128, nlayers::Int=3, 
		init_seed=nothing, prior="normal", pseudoinput_mean=nothing, k=1, kwargs...)

Constructs a classical variational autoencoder.

# Arguments
- `idim::Int`: input dimension.
- `zdim::Int`: latent space dimension.
- `activation::String="relu"`: activation function.
- `hdim::Int=128`: size of hidden dimension.
- `nlayers::Int=3`: number of generator/discriminator layers, must be >= 2. 
- `init_seed=nothing`: seed to initialize weights.
"""
function gan_constructor(;idim::Int=1, zdim::Int=1, activation = "relu", hdim=128, nlayers::Int=2, 
	init_seed=nothing, kwargs...)
	(nlayers < 2) ? error("Less than 3 layers are not supported") : nothing
	
	# if seed is given, set it
	(init_seed != nothing) ? Random.seed!(init_seed) : nothing
	
	# construct the model
	# generator
	generator_map = build_mlp(zdim, hdim, idim, nlayers, activation=activation)
	generator = ConditionalMvNormal(generator_map)
	
	# discriminator
	discriminator_map = build_mlp(idim, hdim, 1, nlayers, activation=activation, lastlayer="σ")
	discriminator = ConditionalMvNormal(discriminator_map)

	# reset seed
	(init_seed != nothing) ? Random.seed!() : nothing

	# constructor form GenerativeModels.jl
	model = GAN(zdim, generator, discriminator)
end

"""
	StatsBase.fit!(model::GenerativeModels.GAN, data::Tuple, gloss:Function, dloss::Function; 
		max_iter=10000, max_train_time=82800, lr=0.001, batchsize=64, patience=30, check_interval::Int=10, 
		kwargs...)
"""
function StatsBase.fit!(model::GenerativeModels.GAN, data::Tuple, gloss::Function, dloss::Function; 
	max_iter=10000, max_train_time=82800, lr=0.001, batchsize=64, patience=30, check_interval::Int=10, 
	kwargs...)
	history = MVHistory()
	dopt = ADAM(lr)
	gopt = ADAM(lr)

	tr_model = deepcopy(model)
	dps = Flux.params(tr_model.discriminator)
	gps = Flux.params(tr_model.generator)
	_patience = patience

	tr_x = data[1][1]
	val_x = data[2][1][:,data[2][2] .== 0]
	val_N = size(val_x,2)

	# on large datasets, batching loss is faster
	best_val_dloss = Inf
	best_val_gloss = Inf
	i = 1
	start_time = time() # end the training loop after 23hrs
	for xbatch in RandomBatches(tr_x, batchsize)
		# disc loss
		batch_dloss = 0f0
		gs = gradient(() -> begin 
			batch_dloss = dloss(tr_model,xbatch)
		end, dps)
	 	Flux.update!(dopt, dps, gs)

	 	# gen loss
		batch_gloss = 0f0
		gs = gradient(() -> begin 
			batch_gloss = gloss(tr_model,xbatch)
		end, gps)
	 	Flux.update!(gopt, gps, gs)

		# only stop if discriminator loss on validation is getting close to 0
		val_dloss = (val_N > 5000) ? dloss(tr_model, val_x, 256) : dloss(tr_model, val_x)
		val_gloss = (val_N > 5000) ? gloss(tr_model, val_x, 256) : gloss(tr_model, val_x)
		(i%check_interval == 0) ? (@info "$i - loss: $(batch_dloss) (dis) $(batch_gloss) (gen)  | validation: $(val_dloss) (dis) $(val_gloss) (gen)") : nothing
		
		# check nans
		if isnan(val_dloss) || isnan(val_gloss) ||  isnan(batch_dloss) || isnan(batch_gloss)
			error("Encountered invalid values in loss function.")
		end

		# save training progress
		push!(history, :training_dloss, i, batch_dloss)
		push!(history, :training_gloss, i, batch_gloss)
		push!(history, :validation_dloss, i, val_dloss)
		push!(history, :validation_gloss, i, val_gloss)
			
		# early stopping
		# only stop if discriminator score gets too close to zero
		if val_dloss > 0.1
			best_val_dloss = val_dloss
			_patience = patience

			# this should save the model at least once
			# when the validation loss is decreasing 
			if mod(i, 10) == 0
				model = deepcopy(tr_model)
			end
		end
		if (time() - start_time > max_train_time) || (i>max_iter) # stop early if time is running out
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
	dloss(model::GenerativeModels.GAN,x[,batchsize])

Classical discriminator loss of the GAN model.
"""
dloss(model::GenerativeModels.GAN,x) = 
	dloss(model.discriminator.mapping,model.generator.mapping,x,rand(model.prior,size(x,ndims(x))))
dloss(model::GenerativeModels.GAN,x,batchsize::Int) = 
	mean(map(y->dloss(m,y), Flux.Data.DataLoader(x, batchsize=batchsize)))

"""
	gloss(model::GenerativeModels.GAN,x[,batchsize])

Classical generator loss of the GAN model.
"""
gloss(model::GenerativeModels.GAN,x) = 
	gloss(model.discriminator.mapping,model.generator.mapping,rand(model.prior,size(x,ndims(x))))
gloss(model::GenerativeModels.GAN,x,batchsize::Int) = 
	mean(map(y->gloss(m,y), Flux.Data.DataLoader(x, batchsize=batchsize)))
	
"""
	generate(model::GenerativeModels.GAN, N::Int)

Generate novel samples.
"""
generate(model::GenerativeModels.GAN, N::Int) = model.generator.mapping(rand(model.prior, N))

"""
	discriminate(model::GenerativeModels.GAN, x)

Discriminate the input - lower score belongs to samples not coming from training distribution.
"""
discriminate(model::GenerativeModels.GAN, x) = model.discriminator.mapping(x)
