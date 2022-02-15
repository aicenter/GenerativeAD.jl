using FileIO, BSON, DataFrames
using ArgParse
using DrWatson
@quickactivate

s = ArgParseSettings()
@add_arg_table s begin
    "fnew"
        help = "the new eval bson"
    "fold"
        help = "the old eval bson"
    "ftarget"
        help = "the target eval bson"   
end
parsed_args = parse_args(ARGS, s)
f_new = datadir("evaluation/$(parsed_args["fnew"])")
f_old = "/home/skvarvit/generativead/GenerativeAD.jl/data/evaluation/$(parsed_args["fold"])"
f_target = datadir("evaluation/$(parsed_args["ftarget"])")

d_new = load(f_new)
d_old = load(f_old)

df = vcat(d_old[:df], d_new[:df])

save(f_target, Dict(:df => df))
