"""
    Implements all needed parts for constructing GANomaly model
        (https://arxiv.org/pdf/1805.06725.pdf) for detecting anomalies.
    Code is inspired by original Pytorch implementation https://github.com/samet-akcay/ganomaly.

"""


struct Encoder
	layers
end

"""
    function ConvEncoder(isize::Int, in_ch::Int, out_ch::Int, nf::Int, extra_layers::Int)

Create the convolutional encoder with parameters
    isize           - size of input image (must be divisible by 16 i.e isize%16==0)
    in_ch           - number of input channels
    out_ch          - number of output channels / dimension of latent space
    nf              - number of covnolutional masks/filters
    extra_layers    - number of additional conv blocks (conv2d + batch norm + leaky relu)
"""
function ConvEncoder(isize::Int, in_ch::Int, out_ch::Int, nf::Int, extra_layers::Int)
    # ověřit jestli isize%16==0
    layers = []
    push!(layers, Conv((4, 4), in_ch => nf; stride = 2, pad = 1))
    push!(layers, BatchNorm(nf))
    push!(layers, x->leakyrelu.(x, 0.2f0))

    csize, cnf = isize/2, nf
    for i=1:extra_layers
        push!(layers, Conv((3, 3), cnf => cnf; stride = 1, pad = 1))
        push!(layers, BatchNorm(cnf))
        push!(layers, x->leakyrelu.(x, 0.2f0))
    end

    while csize > 4
        in_feat = cnf
        out_feat = cnf*2
        push!(layers, Conv((4, 4), in_feat => out_feat; stride = 2, pad = 1))
        push!(layers, BatchNorm(out_feat))
        push!(layers, x->leakyrelu.(x, 0.2f0))
        cnf = cnf*2
        csize=csize/2
    end
    push!(layers, Conv((4, 4), cnf => out_ch; stride = 1, pad = 0))
    return Encoder(Chain(layers...))
end

Flux.@functor Encoder

""" Definition of Encoder's forward pass """
function (e::Encoder)(x)
	e.layers(x)
end



struct Decoder
	layers
end

"""
    function ConvDecoder(isize::Int, in_ch::Int, out_ch::Int, nf::Int, extra_layers::Int)

Create the convolutional decoder with parameters
    isize           - size of output image (must be divisible by 16 i.e isize%16==0)
    in_ch           - number of input channels / dimension of latent space
    out_ch          - number of output channels / same as original input channels
    nf              - number of covnolutional masks/filters
    extra_layers    - number of additional conv blocks (conv2d + batch norm + relu)
"""
function ConvDecoder(isize::Int, in_ch::Int, out_ch::Int, nf::Int, extra_layers::Int)
    layers = []

    cnf, tisize = div(nf,2), 4
    while tisize != isize
        cnf = cnf*2
        tisize = tisize*2
    end
    push!(layers, ConvTranspose((4, 4), in_ch => cnf; stride = 1, pad = 0))
    push!(layers, BatchNorm(cnf, relu))

    csize, _ = 4, cnf
    while csize < div(isize,2)
        push!(layers, ConvTranspose((4, 4), cnf => div(cnf,2); stride = 2, pad = 1))
        push!(layers, BatchNorm(div(cnf,2), relu))
        #push!(layers, x->leakyrelu.(x, 0.2f0))
        cnf = div(cnf,2)
        csize=csize*2
    end
    for i=1:extra_layers
        push!(layers, ConvTranspose((3, 3), cnf => cnf; stride = 1, pad = 1))
        push!(layers, BatchNorm(cnf, relu))
    end
    push!(layers, ConvTranspose((4, 4), cnf => out_ch, tanh; stride = 2, pad = 1))
    return Decoder(Chain(layers...))
end

Flux.@functor Decoder

""" Definition of Decoder's forward pass """
function (d::Decoder)(x)
	d.layers(x)
end


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
    Implementation of Generator, the main core of GANomaly.
    Generator consists of two encoders (E1, E2) and one decoder (D)
"""
struct Generator
    encoder1
    decoder
    encoder2
end

"""
    function Discriminator(isize::Int, latent_dim::Int, in_ch::Int, num_filters::Int, extra_layers::Int)

Create the convolutional discriminator with parameters (almost the same structure as Encoder)
    isize           - size of output image (must be divisible by 16 i.e isize%16==0)
    latent_dim      - dimension of latent space
    in_ch           - number of input (output) channels
    num_filters     - number of covnolutional masks/filters
    extra_layers    - number of additional conv blocks (conv2d + batch norm + leaky relu (resp. relu))
"""
function ConvGenerator(isize, latent_dim, in_ch, num_filters, extra_layers)
    encoder1 = ConvEncoder(isize, in_ch, latent_dim, num_filters, extra_layers)
    decoder = ConvDecoder(isize, latent_dim, in_ch, num_filters, extra_layers)
    encoder2 = ConvEncoder(isize, in_ch, latent_dim, num_filters, extra_layers)
    return Generator(encoder1, decoder, encoder2)
end

Flux.@functor Generator

""" Definition of Generators's forward pass """
function (g::Generator)(x)
	latent_i = g.encoder1(x)
    gen_imag = g.decoder(latent_i)
    latent_o = g.encoder2(gen_imag)
    return gen_imag, latent_i, latent_o
end


"""
    Implementation of loss functions for both generator and discriminator
"""
"""
    function generator_loss(g::Generator, d::Discriminator, real_input, weights=[1,50,1])

will perform forward pass of generator and returns NTuple{4, Float32} of losses.
    Generator loss:     L_gen = w_1 * L_adv + w_2 * L_con + w_3 * L_enc
    Adversarial loss:   L_adv = || f(x) - f(D(E1(x)) ||_2
    Contextual loss:    L_con = || x - D(E1(x)) ||_1
    Encoder loss:       L_enc = || E1(x) - E2(D(E1(x))) ||_2

"""
function generator_loss(g::Generator, d::Discriminator, real_input, weights=[1,50,1])
    fake, latent_i, latent_o = g(real_input)
    pred_real, feat_real = d(real_input)
    pred_fake, feat_fake = d(fake)

    adversarial_loss = Flux.mse(feat_real, feat_fake) # l2_loss
    contextual_loss = Flux.mae(fake, real_input) # l1_loss
    encoder_loss = Flux.mse(latent_o, latent_i)
	return adversarial_loss*weights[1]+contextual_loss*weights[2]+encoder_loss*weights[3],
        adversarial_loss, contextual_loss, encoder_loss
end

"""
    function discriminator_loss(g::Generator, d::Discriminator, real_input)
        &
    function discriminator_loss(d::Discriminator, real_input, fake_input)

are equivalent version functions which perform forward pass of Discriminator
and computes discriminator loss.
"""
function discriminator_loss(g::Generator, d::Discriminator, real_input)
    fake, latent_i, latent_o = g(real_input)
    pred_real, feat_real = d(real_input)
    pred_fake, feat_fake = d(fake)

    loss_for_real = Flux.crossentropy(pred_real, 1f0)
    loss_for_fake = Flux.crossentropy(1f0.-pred_fake, 1f0)
    return 0.5f0*(loss_for_real+loss_for_fake)
end

function discriminator_loss(d::Discriminator, real_input, fake_input)
    pred_real, feat_real = d(real_input)
    pred_fake, feat_fake = d(fake_input)

    loss_for_real = Flux.crossentropy(pred_real, 1f0)
    loss_for_fake = Flux.crossentropy(1f0.-pred_fake, 1f0)
    return 0.5f0*(loss_for_real+loss_for_fake)
end



"""
    Model's training and inference
"""

function StatsBase.fit!(generator::Generator, discriminator::Discriminator, opt, train_loader)

    # training info logger
    history = Dict(
        "generator_loss" => Array{Float32}([]),
        "adversarial_loss" => Array{Float32}([]),
        "contextual_loss" => Array{Float32}([]),
        "encoder_loss" => Array{Float32}([]),
        "discriminator_loss" => Array{Float32}([])
        )

	ps_g = Flux.params(generator)
    ps_d = Flux.params(discriminator)


	progress = Progress(length(train_loader))
	for X in data_loader
		#generator update
		loss1, back = Flux.pullback(ps_g) do
			generator_loss(generator, discriminator, X)
		end
		grad = back((1f0, 0f0, 0f0, 0f0))
		Flux.Optimise.update!(opt, ps_g, grad)

		# discriminator update
		loss2, back = Flux.pullback(ps_d) do
			discriminator_loss(generator, discriminator, X)
		end
		grad = back(1f0)
		Flux.Optimise.update!(opt, ps_d, grad)

		history = update_history(history, loss1, loss2)
		next!(progress; showvalues=[(:generator_loss, loss1[1]), (:discriminator_loss, loss2)])

        #případně přidat kontrolu velikosti chyby s restartem discriminatoru

	end
    return history, generator, discriminator
end

function StatsBase.predict(generator::Generator, discriminator::Discriminator, data)
    anomaly_score = [generator_loss(generator, discriminator, x)[4] for x in Flux.Data.DataLoader(data)]
    s_anomaly_score = (anomaly_score .- minimum(anomaly_score))./(maximum(anomaly_score)-minimum(anomaly_score))
    return s_anomaly_score
end
