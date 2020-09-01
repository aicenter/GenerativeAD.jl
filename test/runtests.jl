using Test
using GenerativeAD
using UCI
using ArgParse
using DrWatson

s = ArgParseSettings()
@add_arg_table! s begin
    "--fast"
    	action = :store_true
        help = "run it fast, i.e. without downloading large datasets"
end
parsed_args = parse_args(ARGS, s)
@unpack fast = parsed_args

include("data.jl")
include("models/runtests.jl")