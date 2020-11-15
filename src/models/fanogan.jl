using Distributions
#using CUDA

"""
	Implements all needed parts for constructing f-AnoGAN model
		(https://www.sciencedirect.com/science/article/pii/S1361841518302640).
"""

struct fAnoGAN
	prior 
	generator
	discriminator
	encoder
end

Flux.@functor fAnoGAN

function Flux.trainable(model::fAnoGAN)
	(generator = model.generator, discriminator = model.discriminator, encoder = model.encoder)
end

function izi(m::fAnoGAN, x)
	z = m.encoder(x)
	x̂ = m.generator(z)
	f = m.discriminator[1:end-1]
	fx = f(x)
	fx̂ = f(x̂)
	return x̂, fx, fx̂
end

"""
	function izif_loss(m::fAnoGAN, x, κ)
loss L_{izi_f} = 1/n * ||x-D(x)||^2 + κ / n_g ||f(x) - f(G(x))||^2
"""
function izif_loss(m::fAnoGAN, x, κ)
	x̂, fx, fx̂ = izi(m, x)
	loss = Flux.mse(x,x̂) + κ .* Flux.mse(fx,fx̂)
	return loss
end

"""
	function anomaly_score(m::fAnoGAN, x, κ=1f0)
Anomaly score computed according to formula from fAnoGAN paper. 
Version izi_f (image-z-image)
		A(x) = 1/n * ||x-D(x)||^2 + κ / n_g ||f(x) - f(G(x))||^2
"""
function anomaly_score(m::fAnoGAN, x, κ=1f0)
	x̂, fx, fx̂ = izi(m, x)
	xx̂= vec(Flux.sum((x̂ .- x).^2, dims=[1,2,3]))
	fxfx̂ = vec(Flux.sum((fx̂ .- fx).^2, dims=1))
	return xx̂ .+ κ .* fxfx̂
end

"""
	function anomaly_score_gpu(m::fAnoGAN, real_input, κ = 1.0; batch_size=64, to_testmode::Bool=true)
Function which compute anomaly score on GPU.
"""
function anomaly_score_gpu(m::fAnoGAN, real_input, κ = 1f0; batch_size=64, to_testmode::Bool=true)
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

# loss functions for Wassersten GAN
wdloss(gen, dis, x, z) = -(Flux.mean(dis(x)) - Flux.mean(dis(gen(z))))
wdloss(model::fAnoGAN, x) = wdloss(model.generator, model.discriminator, x, rand(model.prior, size(x, ndims(x))))
wdloss(model::fAnoGAN, x, z) = wdloss(model.generator, model.discriminator, x, z)

wgloss(gen, dis, z) = - Flux.mean(dis(gen(z)))
wgloss(model::fAnoGAN, z) = wgloss(model.generator, model.discriminator, z)

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

	generator = conv_decoder(idim, zdim, reverse(kernelsizes), reverse(channels), 
		reverse(scalings); activation=activation, batchnorm=batchnorm)

	discriminator = Chain(conv_encoder(idim, 1, kernelsizes, channels, scalings,
			activation=activation, batchnorm=batchnorm)...)
	
	encoder_ =  Chain(conv_encoder(idim, zdim, kernelsizes, channels, scalings,
	activation=activation, batchnorm=batchnorm)..., x->tanh.(x))

	(init_seed !== nothing) ? Random.seed!() : nothing

	model = fAnoGAN(MvNormal(zdim, 1f0), generator, discriminator, encoder_)
end

"""
	function StatsBase.fit!(model::fAnoGAN, data::Tuple, params)

General function for fitting of fAnoGAN.
	model::fAnoGAN
	data::Tupe   		... ((x_train, y_train), (x_valid, y_valid), (x_test, y_test))
	params::NamedTuple  ... all needed parameters for training 
		kappa  		 		... scaling coeficent for izi_f loss function  
		weight_clip  		... weight clipping parameter (https://arxiv.org/abs/1701.07875)
		lr_gan		 		... learning rate for WGAN
		lr_enc		 		... learning rate for encoder (image-z-iamge_f)
		batch_size::Int 	... batch size
		check_every::Int  	... number of iterations between EarlyStopping evaluation check
		patience::Int		... patience before stoping training 
		iters::Int     		... max number of iteration 
		mtt::Int			... max train time (if training is too slow)
		n_critic::Int   	... number of discriminator updater per one generator update
		usegpu::Bool    	... to use GPU or not

	Example: 
		params = (kappa = 1f0, weight_clip=0.1, lr_gan = 0.001, lr_enc = 0.001, batch_size=128, 
			patience=10, check_every=30, iters=10000, mtt=82800, n_critic=1, usegpu=true)	

"""
function StatsBase.fit!(model::fAnoGAN, data::Tuple, params::NamedTuple)

	model = params.usegpu ? model |>gpu : model |>cpu 
	best_model = deepcopy(model)
	history = MVHistory()
	train_loader, val_loader = prepare_dataloaders(data, params)
	val_batches = length(val_loader)
	"""
		WGAN training 
	"""
	
	opt_G = RMSProp(params.lr_gan)
	opt_D = RMSProp(params.lr_gan)
	opt_E = ADAM(params.lr_enc)
	ps_E = Flux.params(model.encoder)
	ps_D = Flux.params(model.discriminator) #model.gan.discriminator
	ps_G = Flux.params(model.generator) # model.gan.generator
	# no early stopping here <– bahavior of loss function will not allow it 
	progress = Progress(length(train_loader))
	for (iter, X) in enumerate(train_loader)
		X = params.usegpu ? gpu(getobs(X)) : getobs(X)
		last_loss_D = 0
		for n = 1:params.n_critic
			z = rand(model.prior, size(X, ndims(X)))
			z = params.usegpu ? z|>gpu : z
			loss_D, back_D = Flux.pullback(ps_D) do
				wdloss(model, X, z)
			end
			grad_D = back_D(1f0)
			Flux.Optimise.update!(opt_D, ps_D, grad_D)
			clip_weights!(ps_D, params.weight_clip)
			last_loss_D = loss_D
		end
		z = rand(model.prior, size(X, ndims(X)))
		z = params.usegpu ? z|>gpu : z
		loss_G, back_G = Flux.pullback(ps_G) do
				wgloss(model, z)
		end
		grad_G = back_G(1f0)
		Flux.Optimise.update!(opt_G, ps_G, grad_G)
		push!(history, :loss_G, iter, loss_G)
		push!(history, :loss_D, iter, last_loss_D)
		next!(progress; showvalues=[
			(:iters, "$(iter)/$(params.iters)"),
			(:loss_G, loss_G),
			(:loss_D, last_loss_D)
			])
		if mod(iter, params.check_every) == 0
			val_loss_G = 0
			val_loss_D = 0
			Flux.testmode!(model)
			for X_val in val_loader
				X_val = params.usegpu ? X_val |> gpu : X_val
				z = rand(model.prior, size(X_val, ndims(X_val)))
				z = params.usegpu ? z|>gpu : z
				val_loss_G += wgloss(model, z)
				val_loss_D += wdloss(model, X_val, z)
			end
			Flux.testmode!(model, false)
			push!(history, :validation_D, iter, val_loss_D/val_batches)
			push!(history, :validation_G, iter, val_loss_G/val_batches)
		end
	end

	"""
		Encoder training
	"""
	# possible memory problems 
	#CUDA.reclaim() # somehow GPU will run out of memory. This helps to clear cached memory and continue with it

	best_model = deepcopy(model)
	# early stopping
	best_val_loss = Inf
	patience_ = deepcopy(params.patience)
	progress = Progress(length(train_loader))

	start_time = time()
	for (iter, X) in enumerate(train_loader)
		X = params.usegpu ? gpu(getobs(X)) : getobs(X)
		loss_E, back_E = Flux.pullback(ps_E) do
			izif_loss(model, X, params.kappa)
		end
		grad_E = back_E(1f0)
		Flux.Optimise.update!(opt_E, ps_E, grad_E)
		
		push!(history, :loss_E, iter, loss_E)
		next!(progress; showvalues=[
			(:iters, "$(iter)/$(params.iters)"),
			(:loss_izif, loss_E)
			])

		if mod(iter, params.check_every) == 0
			val_loss_E = 0
			Flux.testmode!(model)
			for X_val in val_loader
				X_val = params.usegpu ? X_val |> gpu : X_val
				val_loss_E += izif_loss(model, X_val, params.kappa)
			end
			Flux.testmode!(model, false)
			push!(history, :validation_E, iter, val_loss_E/val_batches)
			if val_loss_E < best_val_loss
				best_val_loss = val_loss_E
				patience_ = params.patience
				best_model = deepcopy(model)
			else
				patience_ -= 1
				if patience_ == 0
					@info "Stopped training of Autoencoder after $(iter) iterations"
					break
				end
			end
		end
		if time() - start_time > params.mtt # stop early if time is running out
			best_model = deepcopy(model)
			@info "Stopped training after $(i) iterations, $((time() - start_time)/3600) hours."
			break
		end
	end
	return best_model, history, sum(map(p->length(p), Flux.params(model)))
end
