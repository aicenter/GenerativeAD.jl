module GenerativeAD

using DrWatson
using StatsBase
using Flux
using Statistics
using ImageTransformations
using MLDataPattern

include("datasets/Datasets.jl")
using .Datasets: load_data

include("evaluation/Evaluation.jl")

include("experiments.jl")
include("bayesian_opt.jl")
include("models/Models.jl")

end #module
