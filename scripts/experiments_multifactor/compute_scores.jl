# this is meant to recompute the scores of models on the factorized datasets such as wildlife MNIST
using DrWatson
@quickactivate
using GenerativeAD
using ArgParse
#using Flux, PyCall
include("../pyutils.jl")

s = ArgParseSettings()
@add_arg_table! s begin
	"modelname"
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
    "device"
    	default = "cpu"
    	arg_type = String
    	help = "cpu or cuda"
end
parsed_args = parse_args(ARGS, s)
@unpack modelname, dataset, max_seed, device = parsed_args
method = "leave-one-in"


model = "sgvae"
ac = 1
seed = 1

# load the original train/val/test split
orig_data = GenerativeAD.load_data(dataset, seed=seed, anomaly_class_ind=ac, method=method)
orig_data = GenerativeAD.Datasets.normalize_data(orig_data)
orig_data_denormalized = GenerativeAD.load_data(dataset, seed=seed, anomaly_class_ind=ac, method=method, 
    denormalize=true)


if model in ["sgvae", "cgn"]
	main_modelpath = datadir("sgad_models/images_$(method)/$(modelname)/$(dataset)")
end

# save dir
outdir = datadir("experiments/images_multifactor/$(modelname)/$(dataset)/ac=$(ac)/seed=$(seed)")
mkpath(outdir)

# model dir
modelpath = joinpath(main_modelpath, "ac=$(ac)/seed=$(seed)")
mfs = readdir(modelpath, join=true)
model_ids = map(x->Meta.parse(split(x, "model_id=")[2]), mfs)

# original experiment dir
exppath = datadir("experiments/images_$(method)/$(modelname)/$(dataset)/ac=$(ac)/seed=$(seed)")
expfs = readdir(exppath)

i = 1

mf = mfs[i]
model_id = model_ids[i]

#if "weights" in readdir(mf)
model = load_sgvae_model(mf, device)

expf = filter(x->occursin("$(model_id)", x), expfs)
expdata = 
