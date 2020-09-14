using PyCall
using StatsBase

abstract type PyODmodel end

function StatsBase.fit!(model::PyODmodel, X::Array{T, 2}) where T<:Real
	# transposition since Python models are row major
    model.model.fit(Array(transpose(X)))
end

function StatsBase.predict(model::PyODmodel, X::Array{T, 2}) where T<:Real
	model.model.decision_function(Array(transpose(X)))
end

"""
	LODA(contamination=0.1, n_bins=10, n_random_cuts=100)

The LODA model.
"""
mutable struct LODA <: PyODmodel
	model

	function LODA(;kwargs...)
		py"""
		from pyod.models.loda import LODA

		def construct_loda(kwargs):
			return LODA(**kwargs)
		"""
		new(py"construct_loda"(kwargs))
	end
end

