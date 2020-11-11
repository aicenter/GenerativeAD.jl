using Mill
using StatsBase


"""

    Wrapper for anomalies on groups /bags using decomposition to feature denisty (pf), cardinality (pc) and U 

"""
mutable struct MillModel
    pf::VAE
    pc
    U::Real
end

# TODO! == very weird to copy it around the place!
"""
	loss(model::GenerativeModels.VAE, x[, batchsize])

Negative ELBO for training of a VAE model.
"""
loss(model::GenerativeModels.VAE, x) = -elbo(model, x)
# version of loss for large datasets
loss(model::GenerativeModels.VAE, x, batchsize::Int) = 
	mean(map(y->loss(model,y), Flux.Data.DataLoader(x, batchsize=batchsize)))


"""
    function KNNAnomaly(k::Int, v::Symbol, tree_type::Symbol = :BruteTree)

Create the k-nn anomaly detector with distance variant v::Symbol, where
v = :kappa - the radius of a ball containing all points
v = :gamma - the average distance to all k-nearest neighbors
v = :delta - the average distance to all k-nearest neighbors
"""
# MillModel(pf::VAE, pc, U::Real) =  MillModel(pf,pc,U)


"""
    fit!(model::MillModel, data::Tuple, loss::Function)

    Fit all part of teh mill model: instance density pf, count density, pc, and normalization U
"""
function StatsBase.fit!(model::MillModel, data::Tuple, loss::Function; parameters...) 
    # data are in bagnodes!
    # copy to array
    data_prep = (d)->(d[1].data.data, y_on_instances(d[1],d[2]))
    data_array = (data_prep(data[1]), data_prep(data[2]), data_prep(data[3]))

    info = fit!(model.pf,data_array,loss)

    nofbags = length.(data[1][1].bags)
    model.pc = fit_mle(Distributions.LogNormal, nofbags)

    rlls = raw_ll_score(model.pf, data[1][1]) # raw scores (sum of ll) on train
    model.U = mean(rlls./nofbags)
    (info, model) 
end

"""
    StatsBase.predict(model::KNNAnomaly, x, [k::Int, v::Symbol])

v = :kappa - the radius of a ball containing all points
v = :gamma - the average distance to all k-nearest neighbors
v = :delta - the average distance to all k-nearest neighbors
"""
function StatsBase.predict(model::MillModel)
    if size(model.X,2) == 0
        error("kNN model not fitted, call fit! before predict")
    end
    inds, dists = NearestNeighbors.knn(model.t, x, k, true)
    map(d -> d[end],dists)
end
