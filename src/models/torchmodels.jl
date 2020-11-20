using PyCall
using StatsBase

abstract type TorchModel end

function StatsBase.fit!(model::TorchModel, X::Array{T, 2}, max_iters=10000, lr_gan=1e-4, lr_enc=1e-4, 
    batch_size=64, n_critic=5) where T<:Real
	# transposition since Python models are row major
    model.model.fit(Array(transpose(X)), max_iters, lr_gan, lr_enc, batch_size, n_critic)
end

function StatsBase.predict(model::TorchModel, X::Array{T, 2}) where T<:Real
	model.model.predict(Array(transpose(X)))
end

"""
	fAnoGAN_GP(;idim=(1,2,2), zdim=100, kernelsizes=(9,7,5,3), channels=(128,64,32,16), scalings=(2,2,2,1), 
        activation="relu", batchnorm=false, usegpu=true):)

The fAnoGAN model with gradient penalization.
"""
mutable struct fAnoGAN_GP <: TorchModel
	model

    function fAnoGAN_GP(;kwargs...)
        pushfirst!(PyVector(pyimport("sys")["path"]), "")

		py"""
		import fanogan_gp
        import torch

		def construct_fanogan_gp(kwargs):
            if kwargs.usegpu == True:
                device = torch.device("cuda")
            else:
                device = torch.device("cpu")
			return fanogan_gp()
		"""
		new(py"construct_fanogan_gp"(kwargs))
	end
end