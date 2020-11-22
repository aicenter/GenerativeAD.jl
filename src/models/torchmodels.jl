using PyCall
using StatsBase

pushfirst!(PyVector(pyimport("sys")["path"]), "")
@info "torchmodels compiled. path added -> $(pwd())"
#ushfirst!(PyVector(pyimport("sys")["path"]), @__DIR__)

abstract type TorchModel end

function StatsBase.fit!(model::fAnoGAN_GP, X::Array{T, 4}; max_iters=10000, lr_gan=1e-4, lr_enc=1e-4, 
		batch_size=64, n_critic=5) where T<:Real
	# transposition since Python models are row major
	X = Array(permutedims(X, [4,3,2,1]))
	_, history = model.model.fit(X, max_iters, lr_gan, lr_enc, batch_size, n_critic)
	return model, history
end

function StatsBase.predict(model::fAnoGAN_GP, X::Array{T, 4}) where T<:Real
	X = Array(permutedims(X, [4,3,2,1]))
	return Array(model.model.predict(X))
end
"""
	fAnoGAN_GP(;idim=(1,2,2), zdim=100, kernelsizes=(9,7,5,3), channels=(128,64,32,16), scalings=(2,2,2,1), 
		activation="relu", batchnorm=false, usegpu=true))

The fAnoGAN model with gradient penalization.
"""
mutable struct fAnoGAN_GP <: TorchModel
	model
end

function fAnoGAN_GP(;kwargs...)
py"""
import numpy
import torch
import fanogan_gp

def c_fanogan_gp(kwargs):
	return fanogan_gp.fAnoGAN(**kwargs)
"""
	return fAnoGAN_GP(py"c_fanogan_gp"(kwargs))
end

