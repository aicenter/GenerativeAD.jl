@testset "VAE" begin
	using DrWatson
	@quickactivate
	using Test
	using GenerativeAD
	import StatsBase: fit!, predict
	using ConditionalDists, GenerativeModels, Distributions

	M = 4
	N1 = 100
	N2 = 50
	X = vcat(randn(Float32, Int(M/2), N1), ones(Float32, Int(M/2), N1))
	Y = vcat(randn(Float32, Int(M/2), N2), ones(Float32, Int(M/2), N2))
	Z = vcat(ones(Float32, Int(M/2), N2), randn(Float32, Int(M/2), N2))
	data = ((X,),(Y,))

	parameters = (zdim=8, hdim=32, lr=0.001, batchsize=8, activation="swish", nlayers=3, idim=M)

	model = GenerativeAD.Models.vae_constructor(;parameters...)
	history, iterations, model = fit!(model, data; patience = 100, max_train_time=600, parameters...)
	scores = map(x->GenerativeAD.Models.reconstruction_score(model, x), (X,Y,Z))
	@test mean(scores[1]) < mean(scores[3])
	@test mean(scores[2]) < mean(scores[3])
end