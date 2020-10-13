@testset "GAN" begin
	using DrWatson
	@quickactivate
	using Test
	using GenerativeAD
	import StatsBase: fit!, predict
	using ConditionalDists, GenerativeModels, Distributions
	using EvalMetrics
	using Flux
	import GenerativeAD.Models: discriminate, generate

	# toy example
	M = 4
	N1 = 100
	N2 = 50
	X = vcat(randn(Float32, Int(M/2), N1), ones(Float32, Int(M/2), N1))
	Y = vcat(randn(Float32, Int(M/2), N2), ones(Float32, Int(M/2), N2))
	Z = vcat(ones(Float32, Int(M/2), N2), randn(Float32, Int(M/2), N2))
	data = ((X,),(Y,zeros(size(Y,2))))
	parameters = (zdim=8, hdim=32, lr=0.001, batchsize=8, activation="swish", nlayers=3, idim=M)
	function basic_convergence_test(model, gloss, dloss; kwargs...)
		history, iterations, model = fit!(model, data, gloss, dloss; patience = 100, max_iter=5000,
			max_train_time=600, kwargs...)
		scores = map(x->GenerativeAD.Models.discriminate(model, x), (X,Y,Z))
		@test mean(scores[1]) > mean(scores[3])
		@test mean(scores[2]) > mean(scores[3])
		model
	end
	# vanilla GAN
	model = GenerativeAD.Models.gan_constructor(;parameters...)
	gloss = GenerativeAD.Models.gloss
	dloss = GenerativeAD.Models.dloss
	model = basic_convergence_test(model, gloss, dloss; parameters...)
	@test size(GenerativeAD.Models.generate(model, 10)) == (M,10)
	# feature-matching GAN
	model = GenerativeAD.Models.gan_constructor(;parameters...)
	alpha = 1f0
	fmloss(args...) = alpha*GenerativeAD.Models.gloss(args...) + GenerativeAD.Models.fmloss(args...)
	dloss = GenerativeAD.Models.dloss
	model = basic_convergence_test(model, fmloss, dloss; parameters...)
	# wasserstein gan
	model = GenerativeAD.Models.gan_constructor(;last_linear=true,parameters...)
	dps = Flux.params(model.discriminator)
	wgloss(model,x) = - mean(discriminate(model, generate(model, size(x,ndims(x)))))
	wdloss(model,x) = - mean(discriminate(model,x)) + mean(discriminate(model, generate(model, size(x,ndims(x)))))
	#model = basic_convergence_test(model, wgloss, wdloss; weight_clip=0.01, 
	#	discriminator_advantage=5, stop_threshold=-Inf, max_iter=1500, parameters...)
end
