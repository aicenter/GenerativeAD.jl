using PyCall

abstract type SGADModel end

function StatsBase.fit!(model::SGADModel, X::Array{T, 4}; kwargs...) where T<:Real
    # transposition since Python models are row major
    X = Array(permutedims(X, [4,3,2,1]))
    model.model.train()
    history, _, best_epoch = model.model.fit(X; kwargs...)
    best_model = nothing
    return (history=history, npars=model.model.num_params(), best_model=best_model, best_epoch=best_epoch)
end

function StatsBase.predict(model::SGADModel, X::Array{T, 4}; kwargs...) where T<:Real
    model.model.eval()
    X = Array(permutedims(X, [4,3,2,1]))
    preds = try
        Array(model.model.predict(X; kwargs...))
    catch e
        nothing
    end
    return preds 
end

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

mutable struct SGVAE <: SGADModel
    model
end

function SGVAE(; kwargs...)
    py"""
import sgad
from sgad.sgvae import SGVAE

def SGVAE_constructor(kwargs):
    return SGVAE(**kwargs)
    """

    return SGVAE(py"SGVAE_constructor"(kwargs))
end
