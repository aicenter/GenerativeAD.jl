using PyCall

abstract type SGADModel end

mutable struct CGNAnomaly <: SGADModel
    model
end

function CGNAnomaly(; kwargs...)
    py"""
import sgad
from sgad.cgn.models import CGNAnomaly

def CGNAnomaly_constructor(kwargs):
    return CGNAnomaly(n_classes=1, **kwargs)
    """

    return CGNAnomaly(py"CGNAnomaly_constructor"(kwargs))
end

function StatsBase.fit!(model::CGNAnomaly, X::Array{T, 4}; kwargs...) where T<:Real
    # transposition since Python models are row major
    X = Array(permutedims(X, [4,3,2,1]))
    model.model.train()
    history, _ = model.model.fit(X; kwargs...)
    for k in keys(history)
        history[k] = vcat(history[k]...)
    end
    best_model = nothing
    best_epoch = nothing
    return (history=history, npars=model.model.num_params(), best_model=best_model, best_epoch=best_epoch)
end

function StatsBase.predict(model::CGNAnomaly, X::Array{T, 4}; kwargs...) where T<:Real
    model.model.eval()
    X = Array(permutedims(X, [4,3,2,1]))
    return Array(model.model.predict(X; kwargs...))
end