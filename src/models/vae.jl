####

using Flux
using ConditionalDists
using GenerativeModels

safe_softplus(x) = softplus(x) + 0.000001f0

function vae_constructor(;xdim::Int=1, zdim::Int=1, activation = "relu", hdim=128, nlayers=3)
	(nlayers < 2) ? error("Less than 3 layers are not supported") : nothing
	# function from string
	act = eval(Meta.parse(activation))
	
	# construct the model
	encoder_map = Chain(
		Dense(xdim,hdim,act),
		[Dense(hdim, hdim, act) for _ in 1:nlayers-2]...,
		ConditionalDists.SplitLayer(hdim, [zdim, zdim], [identity, safe_softplus])
		)
	encoder = ConditionalMvNormal(encoder_map)

	decoder_map = Chain(
		Dense(zdim,hdim,act),
		Dense(hdim,hdim,act),
		ConditionalDists.SplitLayer(hdim, [xdim, 1], [identity, safe_softplus])
		)
	decoder = ConditionalMvNormal(decoder_map)

	# get the vanilla VAE
	model = VAE(zdim, encoder, decoder)
end
