module GenerativeAD

using DrWatson
using UCI
using Random
using StatsBase
using DataDeps
using MLDatasets
#using EvalMetrics

include("data.jl")
include("experiments.jl")
#include("evaluation.jl")
include("models/Models.jl")

end #module