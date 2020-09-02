"""

    Implements all three variants of the anomaly detectors based on k-nearest neighbors from
    From outliers to prototypes: Ordering data,
    Stefan Harmelinga and Guido Dornhegea and David Tax and Frank Meinecke and Klaus-Robert Muller, 2005

"""
mutable struct KNNAnomaly{V<:Val}
    t::NNTree
    X::Matrix
    v::V
    k::Int
    tt::Symbol
end

"""
    function KNNAnomaly(k::Int, v::Symbol, tree_type::Symbol = :BruteTree)

Create the k-nn anomaly detector with distance variant v::Symbol, where
v = :kappa - the radius of a ball containing all points
v = :gamma - the average distance to all k-nearest neighbors
v = :delta - the average distance to all k-nearest neighbors
"""
KNNAnomaly(k::Int, v::Symbol, tree_type::Symbol = :BruteTree) = 
    KNNAnomaly(eval(tree_type)(Array{Float32,2}(undef, 1, 0)), Array{Float32,2}(undef, 1, 0), Val(v), k,
        tree_type)


"""
    fit!(model::KNNAnomaly, X::Array{T, 2})
"""
function StatsBase.fit!(model::KNNAnomaly, X::Array{T, 2}) where T<:Real
    model.X = X
    model.t = eval(model.tt)(X)
    return nothing
end

"""
    StatsBase.predict(model::KNNAnomaly, x, [k::Int, v::Symbol])

v = :kappa - the radius of a ball containing all points
v = :gamma - the average distance to all k-nearest neighbors
v = :delta - the average distance to all k-nearest neighbors
"""
function StatsBase.predict(model::KNNAnomaly, x, k, v::V) where {V<:Val{:kappa}}
    if size(model.X,2) == 0
        error("kNN model not fitted, call fit! before predict")
    end
    inds, dists = NearestNeighbors.knn(model.t, x, k, true)
    map(d -> d[end],dists)
end
function StatsBase.predict(model::KNNAnomaly, x, k, v::V) where {V<:Val{:gamma}}
    if size(model.X,2) == 0
        error("kNN model not fitted, call fit! before predict")
    end
    inds, dists = NearestNeighbors.knn(model.t, x, k)
    map(d -> Statistics.mean(d),dists)
end
function StatsBase.predict(model::KNNAnomaly, x, k, v::V) where {V<:Val{:delta}}
    if size(model.X,2) == 0
        error("kNN model not fitted, call fit! before predict")
    end
    inds, dists = NearestNeighbors.knn(model.t, x, k)
    map(i -> LinearAlgebra.norm(x[:,i[1]] - Statistics.mean(model.X[:,i[2]],dims=2)) ,enumerate(inds))
end
StatsBase.predict(model::KNNAnomaly, x, k, v::Symbol) = StatsBase.predict(model, x, k, Val(v))
StatsBase.predict(model::KNNAnomaly, x, k) = StatsBase.predict(model, x, k, model.v)
StatsBase.predict(model::KNNAnomaly, x, v::Symbol) = StatsBase.predict(model, x, model.k, v)
StatsBase.predict(model::KNNAnomaly, x) = StatsBase.predict(model, x, model.k, model.v)

knn_constructor(;k::Int=1,v::Symbol=:kappa) = KNNAnomaly(k, v)