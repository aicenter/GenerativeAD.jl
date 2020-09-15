module Datasets

using UCI
using MLDatasets
using DataDeps
using StatsBase
using DelimitedFiles
using Random

export load_data

include("tabular.jl")
include("img.jl")
include("basics.jl")

end