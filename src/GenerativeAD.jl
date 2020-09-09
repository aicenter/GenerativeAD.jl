module GenerativeAD

using DrWatson
using UCI
using Random
using StatsBase
using DataDeps
using MLDatasets
using Flux
using Statistics
using ImageTransformations
using MLDataPattern
#using EvalMetrics

include("data.jl")
include("utils.jl")
include("experiments.jl")
#include("evaluation.jl")
include("models/Models.jl")
include("models/GANomaly.jl")

end #module
