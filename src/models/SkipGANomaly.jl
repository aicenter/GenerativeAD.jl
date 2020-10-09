"""
	Implements all needed parts for constructing SkipGANomaly model
		(https://arxiv.org/pdf/1901.08954.pdf) for detecting anomalies.
	Code is inspired by original Pytorch implementation https://github.com/samet-akcay/skip-ganomaly.

"""

struct UnetSkipBlock
	model
	cat_pass # BOOL if concatenate input and output "cat(x, model(x))"" or not
end

Flux.@functor UnetSkipBlock

function (Unet::UnetSkipBlock)(x; dims=3)
	if Unet.cat_pass
		return cat(x, Unet.model(x), dims=dims)
	else
		return Unet.model(x)
	end
end

"""
	function ConvUnetSkipBlock(in_ch::Int, inner_ch::Int; sub_level=x->x, layer_type="middle")

Create individual Unet layers which can be stacked.
	in_ch      - number of input channels
	inner_ch   - number of covnolutional masks/filters within this block/layer
	sub_level  - lower Unet block or some function like indentity (x->x)
	layer_type - type of Unet block ["bottom", "middle", "top"]
"""
function ConvUnetSkipBlock(in_ch::Int, inner_ch::Int; sub_level=x->x, layer_type="middle")
	if layer_type == "top"
		pass = false
		fm = 2 # filter multiplier

		down = Chain(Conv((4, 4), in_ch=>inner_ch; stride=2, pad=1), BatchNorm(inner_ch))
		up = Chain(x->relu.(x), ConvTranspose((4, 4), fm*inner_ch=>in_ch; stride=2, pad=1))

		return UnetSkipBlock(Chain(down, sub_level, up), pass)

	elseif layer_type == "bottom"
		pass = true
		fm = 1
	elseif layer_type == "middle"
		pass = true
		fm = 2
	else
		error("unkown layer type")
	end
	down = Chain(x->leakyrelu.(x, 0.2f0),
			Conv((4, 4), in_ch=>inner_ch; stride=2, pad=1),
			BatchNorm(inner_ch))
	up = Chain(x->relu.(x),
			ConvTranspose((4, 4), fm*inner_ch=>in_ch; stride=2, pad=1),
			BatchNorm(in_ch))

	return UnetSkipBlock(Chain(down, sub_level, up), pass)
end


"""
	UnetSkipGenerator
"""

struct UnetSkipGenerator
	model
end

Flux.@functor UnetSkipGenerator

function (Unet::UnetSkipGenerator)(x)
	return Unet.model(x)
end

"""
	function UnetSkipGenerator(isize::Int, in_ch::Int, nf::Int)

Create the convolutional generator (Encoder-Decoder) with skip connections with Unet shape
	isize      - size of output image (must be divisible by 32 i.e isize%32==0)
	in_ch      - number of input channels
	nf         - number of covnolutional masks/filters

Example: isize=32, in_ch=3, nf=64

	3=>64                        ->                     64+64=>3
		|                                                  ^
		v                                                  |
		64=>128                  ->               128+128=>64
			 |                                         ^
			 v                                         |
			 128=>256            ->          256+256=>128
				   |                              ^
				   v                              |
				  256=>512       ->     512+512=>256
						|                 ^
						v                 |
						512=>512 -> 512=>512
"""
function UnetSkipGenerator(isize::Int, in_ch::Int, nf::Int)
	@assert isize%32==0 && isize >= 32
	layers = 0
	while isize >= 2
		layers += 1
		isize = isize/2
	end
	unet = ConvUnetSkipBlock(nf*8, nf*8, layer_type="bottom")
	for i = 1:(layers - 5)
		unet = ConvUnetSkipBlock(nf*8, nf*8, sub_level=unet, layer_type="middle")
	end
	unet = ConvUnetSkipBlock(nf*4, nf*8, sub_level=unet, layer_type="middle")
	unet = ConvUnetSkipBlock(nf*2, nf*4, sub_level=unet, layer_type="middle")
	unet = ConvUnetSkipBlock(nf, nf*2, sub_level=unet, layer_type="middle")
	unet = ConvUnetSkipBlock(in_ch, nf, sub_level=unet, layer_type="top")
	return Chain(UnetSkipGenerator(unet), x->tanh.(x))
end

"""
	SkipGANomaly
"""
struct SkipGANomaly
	generator
	discriminator
end

Flux.@functor SkipGANomaly

function SkipGANomaly(isize::Int, in_ch::Int, out_ch::Int, nf::Int, extra_layers::Int)
	gen = UnetSkipGenerator(isize, in_ch, nf)
	disc = ConvDiscriminator(isize, in_ch, out_ch, nf, extra_layers)
	return SkipGANomaly(gen, disc)
end

"""
	Implementation of loss functions for both generator and discriminator
"""

"""
	function generator_loss(SkipGAN::SkipGANomaly, real_input, weights=[1,50,1])

will perform forward pass of generator and returns NTuple{4, Float32} of losses.
	Generator loss:     L_gen = w_1 * L_adv + w_2 * L_con + w_3 * L_lat
	Adversarial loss:   L_adv = E[log(D(x_real))] + E[log(1-D(x_fake))]
	Contextual loss:    L_con = || x - D(G(x)) ||_1
	Latent loss:        L_lat = || f(x_real) - f(x_fake) ||_2

"""
function generator_loss(SkipGAN::SkipGANomaly, real_input; weights=[1,50,1])
	fake = SkipGAN.generator(real_input) #+ randn(typeof(real_input[1]),size(real_input))

	pred_real, feat_real = SkipGAN.discriminator(real_input)
	pred_fake, feat_fake = SkipGAN.discriminator(fake)

	# adv_loss from original code => self.opt.w_adv * self.l_adv(self.pred_fake, self.real_label)
	adv_loss = Flux.crossentropy(pred_fake, 1f0) # adversarial loss
	con_loss  = Flux.mae(fake, real_input) # contextual loss
	lat_loss = Flux.mse(feat_fake, feat_real) # loss of latent representation

	return adv_loss * weights[1] + con_loss * weights[2] + lat_loss * weights[3],
		adv_loss, con_loss, lat_loss
end

"""
	function discriminator_loss(SkipGAN::SkipGANomaly, real_input)

will perform forward pass of discriminator and returns its loss.
	Discriminator loss:    L_gen = L_adv + L_lat
	Adversarial loss:      L_adv = E[log(D(x_real))] + E[log(1-D(x_fake))]
	Latent loss:           L_lat = || f(x_real) - f(x_fake) ||_2
"""
function discriminator_loss(SkipGAN::SkipGANomaly, real_input)
	fake = SkipGAN.generator(real_input) # + randn(typeof(real_input[1]), size(real_input))

	pred_real, feat_real = SkipGAN.discriminator(real_input)
	pred_fake, feat_fake = SkipGAN.discriminator(fake)

	lat_loss = Flux.mse(feat_fake, feat_real) # loss of latent representation
	loss_for_real = Flux.crossentropy(pred_real, 1f0) # ones(typeof(pred_real), size(pred_real)) has same speed
	loss_for_fake = Flux.crossentropy(1f0.-pred_fake, 1f0)
	return loss_for_real + loss_for_fake + lat_loss
end

"""
	function validation_loss(SkipGAN::SkipGANomaly, real_input, weights=[1,50,1])
computes generator and discriminator loss without additional pass through model
"""
function validation_loss(SkipGAN::SkipGANomaly, real_input; weights=[1,50,1])
	fake = SkipGAN.generator(real_input)

	pred_real, feat_real = SkipGAN.discriminator(real_input)
	pred_fake, feat_fake = SkipGAN.discriminator(fake)

	adv_loss = Flux.crossentropy(pred_fake, 1f0)
	con_loss  = Flux.mae(fake, real_input)
	lat_loss = Flux.mse(feat_fake, feat_real)

	loss_for_real = Flux.crossentropy(pred_real, 1f0)
	loss_for_fake = Flux.crossentropy(1f0.-pred_fake, 1f0)

	return adv_loss * weights[1] + con_loss * weights[2] + lat_loss * weights[3],
		loss_for_real + loss_for_fake + lat_loss, con_loss, lat_loss
end

"""
	function anomaly_score(SkipGAN::SkipGANomaly, real_input; lambda = 0.5)
computes unscaled anomaly score of real input, losses are weighted by factor lambda
"""
function anomaly_score(SkipGAN::SkipGANomaly, real_input; lambda = 0.9, dims=[1,2,3])
	fake = SkipGAN.generator(real_input) # + randn(typeof(real_input[1]), size(real_input))

	pred_real, feat_real = SkipGAN.discriminator(real_input)
	pred_fake, feat_fake = SkipGAN.discriminator(fake)

	rec_loss  = Flux.mae(fake, real_input, agg=x->Flux.mean(x, dims=dims)) # reconstruction loss -> similar to contextual loss
	lat_loss = Flux.mse(feat_fake, feat_real, agg=x->Flux.mean(x, dims=dims)) # loss of latent representation
	return lambda .* rec_loss .+ (1 - lambda) .* lat_loss |>cpu
end

function generalized_anomaly_score(SkipGAN::SkipGANomaly, real_input; R="mae", L="mae", lambda=0.9, dims=[1,2,3])
	fake = SkipGAN.generator(real_input) # + randn(typeof(real_input[1]), size(real_input))

	pred_real, feat_real = SkipGAN.discriminator(real_input)
	pred_fake, feat_fake = SkipGAN.discriminator(fake)

	R = getfield(Flux, Symbol(R))
	L = getfield(Flux, Symbol(L))
	rec_loss  = R(fake, real_input, agg=x->Flux.mean(x, dims=dims)) # reconstruction loss -> similar to contextual loss
	lat_loss = L(feat_fake, feat_real, agg=x->Flux.mean(x, dims=dims)) # loss of latent representation
	return lambda .* rec_loss .+ (1 - lambda) .* lat_loss |>cpu
end

function generalized_anomaly_score_gpu(SkipGAN::SkipGANomaly, real_input; R="mae", L="mae", lambda=0.9, dims=[1,2,3], batch_size=64)
	real_input = Flux.Data.DataLoader(real_input, batchsize=batch_size)
	SkipGAN = SkipGAN |> gpu
	R = getfield(Flux, Symbol(R))
	L = getfield(Flux, Symbol(L))
	Rs = Array{Float32}([])
	Ls = Array{Float32}([])
	for X in real_input
		fake = SkipGAN.generator(X|>gpu)
		pred_real, feat_real = SkipGAN.discriminator(X|>gpu)
		pred_fake, feat_fake = SkipGAN.discriminator(fake)

		Rs = cat(Rs, vec(R(fake, X|>gpu, agg=x->Flux.mean(x, dims=dims))|>cpu), dims=1)
		Ls = cat(Ls, vec(L(feat_fake, feat_real, agg=x->Flux.mean(x, dims=dims))|>cpu), dims=1)
		#output = cat(output,vec(Flux.mae(latent_i, latent_o, agg=x->mean(x, dims=dims))) |> cpu, dims=1)
	end
	return lambda .* Rs .+ (1 - lambda) .* Ls
end

"""
	Model's training and inference
"""

"""
	fit!(SkipGAN::SkipGANomaly, data, params)
"""
function StatsBase.fit!(SkipGAN::SkipGANomaly, data, params)
	# prepare batches & loaders
	train_loader, val_loader = prepare_dataloaders(data, params)
	# training info logger
	history = GANomalyHistory()
	history["anomality"] =  Array{Float32}([])
	# prepare for early stopping
	best_model = deepcopy(SkipGAN)
	patience = params.patience
	best_val_loss = 1e10
	val_batches = length(val_loader)
	# define optimiser
	# ADAMW(η = 0.001, β = (0.9, 0.999), decay = 0) = Optimiser(ADAM(η, β), WeightDecay(decay))
	opt = haskey(params, :decay) ? ADAMW(params.lr, (0.9, 0.999), params.decay) : ADAM(params.lr)

	ps_g = Flux.params(SkipGAN.generator)
	ps_d = Flux.params(SkipGAN.discriminator)

	progress = Progress(length(train_loader))
	for (iter, X) in enumerate(train_loader)
		#generator update
		loss1, back = Flux.pullback(ps_g) do
			generator_loss(SkipGAN, getobs(X)|>gpu, weights=params.weights)
		end
		grad = back((1f0, 0f0, 0f0, 0f0))
		Flux.Optimise.update!(opt, ps_g, grad)

		# discriminator update
		loss2, back = Flux.pullback(ps_d) do
			discriminator_loss(SkipGAN, getobs(X)|>gpu)
		end
		grad = back(1f0)
		Flux.Optimise.update!(opt, ps_d, grad)

		history = update_history(history, loss1, loss2)
		next!(progress; showvalues=[
			(:iters, "$(iter)/$(params.iters)"),
			(:generator_loss, loss1[1]),
			(:discriminator_loss, loss2)
			])
		#TODO optionaly add discriminator restrart if its loss drops under 1e-5
		if mod(iter, params.check_every) == 0
			tot_val_loss_g, tot_val_loss_d, tot_val_loss_rl, tot_val_loss_ll = 0, 0, 0, 0
			for X_val in val_loader
				vgl, vdl,  rl, ll = validation_loss(SkipGAN, X_val |> gpu, weights=params.weights)
				tot_val_loss_g += vgl
				tot_val_loss_d += vdl
				tot_val_loss_rl += rl
				tot_val_loss_ll += ll
			end
			history = update_val_history(history, tot_val_loss_g/val_batches, tot_val_loss_d/val_batches)
			anomality = (params.lambda*tot_val_loss_rl + (1-params.lambda)*tot_val_loss_ll)
			push!(history["anomality"], anomality)
			if anomality < best_val_loss
				best_val_loss = anomality
				patience = params.patience
				best_model = deepcopy(SkipGAN)
			else
				patience -= 1
				if patience == 0
					@info "Stopped training after $(iter) iters"
					global iters = iter - params.check_every*params.patience
					break
				end
			end
		end
		if iter == params.iters
			global iters = params.iters
		end
	end
	return history, best_model, sum(map(p->length(p), ps_g)) + sum(map(p->length(p), ps_d)), iters
end

"""
	conv SkipGANomaly constructor for parameters
"""
function SkipGANomaly_constructor(kwargs)
	params = (isize=kwargs.isize,
			  in_ch = kwargs.in_ch,
			  out_ch = kwargs.out_ch,
			  nf = kwargs.num_filters,
			  extra_layers = kwargs.extra_layers)

	# little control to random initialization
  	(kwargs.init_seed != nothing) ? Random.seed!(kwargs.init_seed) : nothing
	SkipGAN = SkipGANomaly(params...)
	(kwargs.init_seed != nothing) ? Random.seed!() : nothing
	return SkipGAN, params
end
