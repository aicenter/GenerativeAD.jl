using PyCall
using StatsBase

mutable struct PIDForest
    forest

	function PIDForest(parameters) 
		py"""
		from pidforest.forest import Forest

		def construct_pidforest(kwargs):
			return Forest(**kwargs)
		"""
		new(py"construct_pidforest"(parameters))
	end
end

function StatsBase.fit!(model::PIDForest, X::Array{T, 2}) where T<:Real
	# no need for data transposition as the forest is learned on column major
    model.forest.fit(X)
end

function StatsBase.predict(model::PIDForest, x; pct=50) where T<:Real
	# anomaly scores correspond to percentile `pct` over trees
	# the lower the score the more anomalous the sample should be
    -model.forest.predict(x; pct=pct)[end]
end
