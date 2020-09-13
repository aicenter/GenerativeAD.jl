abstract type SKmodel end

function StatsBase.fit!(model::SKmodel, X::Array{T, 2}) where T<:Real
	# transposition since Python models are row major
    model.model.fit(Array(transpose(X)))
end

function StatsBase.predict(model::SKmodel, X::Array{T, 2}) where T<:Real
	# anomaly scores correspond to percentile `pct` over trees
    -model.model.score_samples(Array(transpose(X)))
end

mutable struct LOF <: SKmodel
	model

	function LOF(;kwargs...)
		py"""
		from sklearn.neighbors import LocalOutlierFactor

		def construct_lof(kwargs):
			kwargs['novelty'] = True
			kwargs['contamination'] = "auto"
			return LocalOutlierFactor(**kwargs)
		"""
		new(py"construct_lof"(kwargs))
	end
end

mutable struct OCSVM <: SKmodel
	model

	function OCSVM(;kwargs...)
		py"""
		from sklearn.svm import OneClassSVM

		def construct_ocsvm(kwargs):
			return OneClassSVM(**kwargs)
		"""
		new(py"construct_ocsvm"(kwargs))
	end
end
# TODO add kernel type to params

mutable struct IForest <: SKmodel
	model

	function IForest(;kwargs...)
		py"""
		from sklearn.ensemble import IsolationForest

		def construct_if(kwargs):
			kwargs['contamination'] = "auto"
			kwargs['behaviour'] = "new"
			return IsolationForest(**kwargs)
		"""
		new(py"construct_if"(kwargs))
	end
end
