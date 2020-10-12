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
	data = ((X,),(Y,zeros(size(Y,2))))
	# first test conv_encoder and decoder
	idim = size(X)[1:3]
	zdim = 4
	kernelsizes = (3,5)
	channels = (2,4)
	scalings = (2,1)
	encoder = GenerativeAD.Models.conv_encoder(idim, zdim, kernelsizes, channels, scalings)
	@test size(encoder(X)) == (zdim, N1)
	decoder = GenerativeAD.Models.conv_decoder(idim, zdim, kernelsizes, channels, scalings)
	z = randn(zdim, N1)
	@test size(decoder(z)) == (idim..., N1)

	
end
