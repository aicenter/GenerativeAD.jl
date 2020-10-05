using DrWatson
@quickactivate
using Test
using GenerativeAD
using UCI
using ArgParse

s = ArgParseSettings()
@add_arg_table! s begin
    "--complete"
    	action = :store_true
        help = "run a more thorough set of tests, may be slow"
end
parsed_args = parse_args(ARGS, s)
@unpack complete = parsed_args

include("data.jl")
include("experiments.jl")
include("models/runtests.jl")