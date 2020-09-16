module Datasets

using UCI
using MLDatasets
using DataDeps
using StatsBase
using DelimitedFiles
using Random
using CSV
using DataFrames
using Flux # for one-hot encoding

export load_data

include("tabular.jl")
include("img.jl")
include("basics.jl")

end