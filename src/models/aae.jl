struct AAE
	prior
	encoder
	decoder
	discriminator
end

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
	discriminator = build_mlp(zdim, hdim, 1, nlayers, activation=activation, lastlayer="softplus")

	# prior
	if prior == "normal"
		W = first(Flux.params(encoder))
	    μ = fill!(similar(W, zlength), 0)
	    σ = fill!(similar(W, zlength), 1)
	    prior = DistributionsAD.TuringMvNormal(μ, σ)
	elseif prior == "vamp"
		(pseudoinput_mean == nothing) ? error("if `prior=vamp`, supply pseudoinput array") : nothing
		T = eltype(pseudoinput_mean)
		pseudoinputs = T(1) .* randn(T, size(pseudoinput_mean)[1:end-1]..., k) .+ pseudoinput_mean
		prior = VAMP(pseudoinputs)
	end

	# reset seed
	(init_seed != nothing) ? Random.seed!() : nothing

	# get the vanilla VAE
	model = AAE(prior, encoder, decoder, discriminator)
end
