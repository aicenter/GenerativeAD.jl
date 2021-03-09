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
using NPZ
using Images

export load_data

include("datadeps_init.jl")
include("tabular.jl")
include("img.jl")
include("basics.jl")

end