module Models

using NearestNeighbors
using StatsBase
using Statistics
using LinearAlgebra

using Flux
using ImageTransformations
using MLDataPattern
using ProgressMeter: Progress, next!

include("utils/losses.jl")
include("utils/nn_builders.jl")

include("knn.jl")
include("pidforest.jl")
include("GANomaly.jl")
include("skmodels.jl")
include("pyodmodels.jl")
include("real_nvp.jl")


end
