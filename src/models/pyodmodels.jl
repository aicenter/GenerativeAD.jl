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

"""
	ABOD(n_neighbors=5, method='fast')

The ABOD model.
"""
mutable struct ABOD <: PyODmodel
	model

	function ABOD(;kwargs...)
		py"""
		from pyod.models.abod import ABOD

		def construct_abod(kwargs):
			return ABOD(**kwargs)
		"""
		new(py"construct_abod"(kwargs))
	end
end


"""
	HBOS(n_bins=10, alpha=0.1, tol=0.5)

The HBOS model.
"""
mutable struct HBOS <: PyODmodel
	model

	function HBOS(;kwargs...)
		py"""
		from pyod.models.hbos import HBOS

		def construct_hbos(kwargs):
			return HBOS(**kwargs)
		"""
		new(py"construct_hbos"(kwargs))
	end
end


"""
	MO_GAAL(k=10, stop_epochs=20, lr_d=0.01, lr_g=0.0001, decay=1e-06, momentum=0.9, contamination=0.1)

The MO_GAAL model. (GAN based)
"""
mutable struct MO_GAAL <: PyODmodel
	model

	function MO_GAAL(;kwargs...)
		py"""
		from pyod.models.mo_gaal import MO_GAAL

		def construct_mogaal(kwargs):
			return MO_GAAL(**kwargs)
		"""
		new(py"construct_mogaal"(kwargs))
	end
end