# this is meant to recompute the scores of models on the factorized datasets such as wildlife MNIST
using DrWatson
@quickactivate
using GenerativeAD
using ArgParse
#using Flux, PyCall
include("../pyutils.jl")

s = ArgParseSettings()
@add_arg_table! s begin
	"model"
		arg_type = String
		help = "model name"
    "dataset"
        default = "wildlife_MNIST"
        arg_type = String
        help = "dataset"
  	"max_seed"
        default = 5
        arg_type = Int
        help = "seed"
end
parsed_args = parse_args(ARGS, s)
@unpack model, dataset, max_seed = parsed_args
method = "leave-one-in"


model = "sgvae"
ac = 1
seed = 1

if model in ["sgvae", "cgn"]
	main_modelpath = datadir("sgad_models/images_$(method)/$(model)/$(dataset)")
end

modelpath = joinpath(main_modelpath, "ac=$(ac)/seed=$(seed)")
mfs = readdir(modelpath, join=true)
model_ids = map(x->Meta.parse(split(x, "model_id=")[2]), mfs)

i = 1

mf = mfs[i]
model_id = model_ids[i]

if "weights" in readdir(mf)
