@testset "ConvVAE" begin
	using DrWatson
	@quickactivate
	using Test
	using GenerativeAD
	import StatsBase: fit!, predict
	using ConditionalDists, GenerativeModels, Distributions
	using EvalMetrics
	using IPMeasures
	using Flux

	
	# toy example
	M = (8,8,1)
	M2 = (4,8,1)
	N1 = 100
	N2 = 50
	X = cat(randn(Float32, M2..., N1), ones(Float32, M2..., N1), dims=1)
	Y = cat(randn(Float32, M2..., N2), ones(Float32, M2..., N2), dims=1)
	Z = cat(ones(Float32, M2..., N2), randn(Float32, M2..., N2), dims=1)
	data = ((X,),(Y,zeros(size(Y,4))))

	# first test conv_encoder and decoder
	idim = size(X)[1:3]
	zdim = 4
	kernelsizes = (3,5)
	channels = (2,4)
	scalings = (2,1)
	encoder = GenerativeAD.Models.conv_encoder(idim, zdim, kernelsizes, channels, scalings)
	@test size(encoder(X)) == (zdim, N1)
	decoder = GenerativeAD.Models.conv_decoder(idim, zdim, reverse(kernelsizes), 
		reverse(channels), reverse(scalings))
	z = randn(zdim, N1)
	@test size(decoder(z)) == (idim..., N1)
	hdim = 32
	decoder = GenerativeAD.Models.conv_decoder(idim, zdim, reverse(kernelsizes), reverse(channels), 
		reverse(scalings), activation="tanh", vec_output=true, vec_output_dim=hdim)
	@test size(decoder(z)) == (hdim, N1)
	function basic_convergence_test(model, loss, scoref; kwargs...)
		history, iterations, model = fit!(model, data, loss; patience = 100, max_train_time=600, 
			kwargs...)
		scores = map(x->scoref(model, x), (X,Y,Z))
		@test mean(scores[1]) < mean(scores[3])
		@test mean(scores[2]) < mean(scores[3])
		model
	end
	# test convolutional vae
	parameters = (idim=idim, zdim=zdim, activation="swish", hdim=32, kernelsizes=(5,3),
		channels=(4,8), scalings=(2,2))
	model = GenerativeAD.Models.conv_vae_constructor(;parameters...)
	loss(m,x) = -elbo(m,x)
	loss(m, x, batchsize::Int) = 
		mean(map(y->loss(m,y), Flux.Data.DataLoader(x, batchsize=batchsize)))
	model = basic_convergence_test(model, loss, GenerativeAD.Models.reconstruction_score)
	@test size(GenerativeAD.Models.reconstruct(model, X)) == size(X)
	@test size(GenerativeAD.Models.generate(model, 10, idim)) == (idim..., 10)
	# gpu
	gmodel = GenerativeAD.Models.conv_vae_constructor(;parameters...) |> gpu
	gX = X |> gpu
	# the Array is there since gpu computations dont work on views created by MLDataPattern iterators
	gloss(m,x) = -elbo(m,gpu(Array(x))) 
	gloss(m, x, batchsize::Int) = 
		mean(map(y->gloss(m,y), Flux.Data.DataLoader(x, batchsize=batchsize)))
	gscoref(model, x) = GenerativeAD.Models.reconstruction_score(model, gpu(x))	
	gmodel = basic_convergence_test(gmodel, gloss, gscoref; parameters...)
end
