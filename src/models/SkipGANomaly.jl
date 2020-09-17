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
    Discriminator
"""

struct Discriminator
	features
	classifier
end
"""
	function Discriminator(isize::Int, in_ch::Int, out_ch::Int, nf::Int, extra_layers::Int)

Create the convolutional discriminator with parameters (almost the same structure as Encoder)
	isize           - size of output image (must be divisible by 16 i.e isize%16==0)
	in_ch           - number of input channels / dimension of latent space
	out_ch          - number of output channels / same as original input channels
	nf              - number of covnolutional masks/filters
	extra_layers    - number of additional conv blocks (conv2d + batch norm + leaky relu)
"""
function ConvDiscriminator(isize::Int, in_ch::Int, out_ch::Int, nf::Int, extra_layers::Int)
	features = []
	push!(features, Conv((4, 4), in_ch => nf; stride = 2, pad = 1))
	push!(features, BatchNorm(nf))
	push!(features, x->leakyrelu.(x, 0.2f0))

	csize, cnf = isize/2, nf
	for i=1:extra_layers
		push!(features, Conv((3, 3), cnf => cnf; stride = 1, pad = 1))
		push!(features, BatchNorm(cnf))
		push!(features, x->leakyrelu.(x, 0.2f0))
	end

	while csize > 4
		in_feat = cnf
		out_feat = cnf*2
		push!(features, Conv((4, 4), in_feat => out_feat; stride = 2, pad = 1))
		push!(features, BatchNorm(out_feat))
		push!(features, x->leakyrelu.(x, 0.2f0))
		cnf = cnf*2
		csize=csize/2
	end
	cls = Conv((4, 4), cnf => out_ch, sigmoid; stride = 1, pad = 0)
	return Discriminator(Chain(features...), cls)
end

Flux.@functor Discriminator

""" Definition of Discriminator's forward pass """
function (d::Discriminator)(x)
	feat = d.features(x)
	class = d.classifier(feat)
	return class, feat
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
	pred_fake, feat_fake = SkipGAN.discriminator(fake_input)

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
        loss_for_real + loss_for_fake + lat_loss
end

"""
    function anomaly_score(SkipGAN::SkipGANomaly, real_input; lambda = 0.5)
computes unscaled anomaly score of real input, losses are weighted by factor lambda
"""
function anomaly_score(SkipGAN::SkipGANomaly, real_input; lambda = 0.9)
    fake = SkipGAN.generator(real_input) # + randn(typeof(real_input[1]), size(real_input))

    pred_real, feat_real = SkipGAN.discriminator(real_input)
    pred_fake, feat_fake = SkipGAN.discriminator(fake_input)

    rec_loss  = Flux.mse(fake, real_input) # reconstruction loss -> similar to contextual loss
    lat_loss = Flux.mse(feat_fake, feat_real) # loss of latent representation
	return lambda * rec_loss + (1 - lambda) * lat_loss

end


"""
	Model's training and inference
"""

"""
	fit!(SkipGAN::SkipGANomaly, data, params)
"""
function StatsBase.fit!(SkipGAN::SkipGANomaly, data, params)
    # prepare batches & loaders
    train_loader, val_loader = GenerativeAD.prepare_dataloaders(data, params)
    # training info logger
    history = GenerativeAD.GANomalyHistory()
    # prepare for early stopping
    best_model = deepcopy(SkipGAN)
    patience = params.patience
    best_val_loss = 1e10
    # define optimiser
	opt = Flux.Optimise.ADAM(params.lr)

	ps_g = Flux.params(SkipGAN.generator)
	ps_d = Flux.params(SkipGAN.discriminator)

	for epoch = 1:params.epochs
		progress = Progress(length(train_loader))
		for X in train_loader
			#generator update
			loss1, back = Flux.pullback(ps_g) do
				generator_loss(SkipGAN, X|>gpu, weights=params.weights)
			end
			grad = back((1f0, 0f0, 0f0, 0f0))
			Flux.Optimise.update!(opt, ps_g, grad)

			# discriminator update
			loss2, back = Flux.pullback(ps_d) do
				discriminator_loss(SkipGAN, X|>gpu)
			end
			grad = back(1f0)
			Flux.Optimise.update!(opt, ps_d, grad)

			history = GenerativeAD.update_history(history, loss1, loss2)
			next!(progress; showvalues=[(:epoch, "$(epoch)/$(params.epochs)"),
										(:generator_loss, loss1[1]),
										(:discriminator_loss, loss2)
										])
			#TODO optionaly add discriminator restrart if its loss drops under 1e-5
		end
        total_val_loss_g = 0
        total_val_loss_d = 0
        for X_val in val_loader
            vgl, vdl = validation_loss(SkipGAN, X_val |> gpu, weights=params.weights)
            total_val_loss_g += vgl
            total_val_loss_d += vdl
        end
        history = GenerativeAD.update_history(history, nothing, nothing, total_val_loss_g total_val_loss_d)
        if total_val_loss_g < best_val_loss
            best_val_loss = total_val_loss_g
            patience = params.patience
            best_model = deepcopy(SkipGAN)
        else
            patience -= 1
            if patience == 0
                @info "Stopped training after $(epoch) epochs"
                break
            end
        end
	end
	return history, best_model
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

    SkipGAN = SkipGANomaly(params...)
	return SkipGAN, params
end