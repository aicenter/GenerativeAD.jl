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
    model.forest.fit(X)
end

function StatsBase.predict(model::PIDForest, x; err=0.1, pct=50) where T<:Real
    -model.forest.predict(x; err=0.1, pct=50)[end] # last element contains scores
end

