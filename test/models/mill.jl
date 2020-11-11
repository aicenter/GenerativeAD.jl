@testset "Mill VAE" begin
	using DrWatson
	@quickactivate
	using Test
	using GenerativeAD
	import StatsBase: fit!, predict
	using ConditionalDists, GenerativeModels, Distributions
	using EvalMetrics
    using Flux
    using Mill
	include(srcdir()*"/datasets/mill.jl")
	include(srcdir()*"/models/mill.jl")
	include(srcdir()*"/models/utils/mill_utils.jl")
	# toy example
	M = 3
	N1 = 3
	Nb = 1
    B = BagNode(ArrayNode(rand(M,N1)),[1:N1÷3,(N1÷3+1):N1÷3*2,(N1÷3*2+1):N1])
    data = ((B[1:2],[0,1]),(B[3],[0]),(B[3],[0]))

	parameters = (zdim=8, hdim=32, lr=0.001, batchsize=1, activation="swish", nlayers=3, idim=M)
	# vanilla VAE
    vae = GenerativeAD.Models.vae_constructor(;parameters...)
	loss(model::GenerativeModels.VAE, x) = -elbo(model, x)


    data_vae=((rand(M,Nb),rand(0:1,Nb)),(rand(M,Nb),rand(0:1,Nb)))
    history_vae, iterations_vae, v = fit!(vae,data_vae, loss; patience = 100, max_train_time=600, parameters...)


	raw_ll_score(vae,)
	model =MillModel(vae,nothing,0.0)    
	
	data_prep = (d)->(d[1].data.data, y_on_instances(d[1],d[2]))
    data_array = (data_prep(data[1]), data_prep(data[2]), data_prep(data[3]))

	history, model = fit!(model, data, loss; patience = 100, max_train_time=600)


end
