"""
	Implements all needed parts for constructing f-AnoGAN model
		(https://www.sciencedirect.com/science/article/pii/S1361841518302640).
"""
using Distributions
using DistributionsAD
using ConditionalDists
using GenerativeModels


struct fAnoGAN
	gan::GenerativeModels.GAN
	encoder
end

Flux.@functor fAnoGAN

function izi(m::fAnoGAN, x)
	z = m.encoder(x)
	x̂ = m.generator.mapping(z)
	f = m.discriminator.mapping[1:end-1]
	fx = f(x)
	fx̂ = f(x̂)
	return x̂, fx, fx̂
end

function izif_loss(m::fAnoGAN, x, κ)
	x̂, fx, fx̂ = izi(m, x)
	loss = Flux.mse(x,x̂) + κ .* Flux.mse(fx,fx̂)
	return loss
end

function anomaly_score(m::fAnoGAN, x, κ=1f0)
	x̂, fx, fx̂ = izi(m, x)
	xx̂= vec(Flux.sum((x̂ .- x).^2, dims=[1,2,3]))
	fxfx̂ = vec(Flux.sum((fx̂ .- fx).^2, dims=1))
	return xx̂ .+ κ .* fxfx̂
end

function anomaly_score_gpu(m::fAnoGAN, real_input, κ = 1.0; batch_size=64, to_testmode::Bool=true)
	real_input = Flux.Data.DataLoader(real_input, batchsize=batch_size)
	(to_testmode == true) ? Flux.testmode!(m) : nothing
	m = m |> gpu
	scores = Array{Float32}([])
	for x in real_input
		x = x |> gpu
		score = anomaly_score(m, x, κ)
		scores = cat(scores, score |> cpu, dims=1)
	end
	(to_testmode == true) ? Flux.testmode!(m, false) : nothing
	return scores
end


function fanogan_constructor(
		;idim=(2,2,1), 
		zdim::Int=1, 
		activation="relu", 
		hdim=1024, 
		kernelsizes=(1,1), 
		channels=(1,1), 
		scalings=(1,1),
		init_seed=nothing,	
		batchnorm=false, 
		kwargs...
	)

	(init_seed !== nothing) ? Random.seed!(init_seed) : nothing

	generator_map = conv_decoder(idim, zdim, reverse(kernelsizes), reverse(channels), 
		reverse(scalings); activation=activation, batchnorm=batchnorm)
	generator = ConditionalMvNormal(generator_map)

	vecdim = reduce(*,idim[1:3]) # size of vectorized data
	discriminator_map = Chain(conv_encoder(idim, 1, kernelsizes, channels, scalings,
			activation=activation, batchnorm=batchnorm)..., x->σ.(x))
	discriminator = ConditionalMvNormal(discriminator_map)
	
	encoder_ =  Chain(conv_encoder(idim, zdim, kernelsizes, channels, scalings,
	activation=activation, batchnorm=batchnorm)..., x->tanh.(x))

	(init_seed !== nothing) ? Random.seed!() : nothing

	model = fAnoGAN(GAN(zdim, generator, discriminator), encoder_)
end


function StatsBase.fit!(
		model::fAnoGAN, 
		data::Tuple, 
		kappa = 1f0,
		weight_clip=nothing, # wasserstion 
		max_iter=10000, 
		mtt_gan=82800, 
		mtt_encoder = 82800,
		lr_gan=0.001, 
		lr_enc=0.001
		batchsize=64, 
		patience::Int=10, 
		check_every::Int=30, 
		discriminator_advantage::Int=1, 
		stop_threshold=0.01, 
		usegpu=false,
		kwargs...
	)

	# train WGAN
	gan_info =  fit!(
					model.gan, data, gloss, dloss; max_iter = max_iter, max_train_time=mtt_gan, lr=lr_gan, batchsize=batchsize, 
					patience=patience, check_interval=check_every, weight_clip=weight_clip, stop_threshold=stop_threshold,
					discriminator_advantage::Int=discriminator_advantage,  usegpu=usegpu, kwargs...
				)

	model = fAnoGAN(gan_info.model, model.encoder) # we are getting best model
	best_model = deepcopy(model)

	history = MVHistory()
	train_loader, val_loader = prepare_dataloaders(data, batch_size=batchsize, iters=max_iter)
	best_val_loss = Inf
	patience_ = deepcopy(patience)
	val_batches = length(val_loader)

	model = usegpu ? model|>gpu : model|>cpu
	optim = ADAM(lr_enc)

	start_time = time()
	ps = Flux.params(model.encoder)
	for (iter, X) in enumerate(train_loader)
		X = usegpu ? getobs(X)|>gpu : getobs(X)
		loss_e, back = Flux.pullback(ps) do
			izif_loss(m::fAnoGAN, X, kappa)
		end
		grad = back(1f0)
		Flux.Optimise.update!(optim, ps, grad)
		
		push!(history, :loss_izif, iter, loss_e)

		next!(progress; showvalues=[
			(:iters, "$(iter)/$(max_iter)"),
			(:loss_izif, loss_e)
			])

		if mod(iter, check_every) == 0
			total_val_loss = 0
			Flux.testmode!(model)
			for X_val in val_loader
				X_val = usegpu ? X_val |> gpu : X_val
				total_val_loss += izif_loss(m::fAnoGAN, X, kappa)
			end
			Flux.testmode!(model, false)
			push!(history, :validation_izif, iter, total_val_loss/val_batches)
			if total_val_loss < best_val_loss
				best_val_loss = total_val_loss
				patience_ = patience
				best_model = deepcopy(model)
			else
				patience_ -= 1
				if patience_ == 0
					@info "Stopped training of Autoencoder after $(iter) iterations"
					break
				end
			end
		end
		if time() - start_time > mtt_encoder+mtt_gan # stop early if time is running out
			best_model = deepcopy(model)
			@info "Stopped training after $(i) iterations, $((time() - start_time)/3600) hours."
			break
		end
	end
	return best_model, (gan_history = gan_info.history, izif_history = history, 
		iters_gan = gan_info.iterations, 
		iters_enc = iter, 
		npars=sum(map(p->length(p), Flux.params(model))))
end
