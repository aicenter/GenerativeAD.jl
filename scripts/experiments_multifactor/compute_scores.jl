# this is meant to recompute the scores of models on the factorized datasets such as wildlife MNIST
using DrWatson
@quickactivate
using GenerativeAD
using ArgParse
using DataFrames
using StatsBase
using Dates
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

# this is to understand whether to normalize data or not for sgvae - see commit
# 3ad577956bc7b8690e0a4ab63d7d6fb4d26a0819 in GenerativeAD.jl#sgad
#norm_date = DateTime(2022,3,31)
#norm_time = Dates.datetime2unix(norm_date)

modelname = "sgvae"
ac = 1
seed = 1

# load the original train/val/test split
orig_data = GenerativeAD.load_data(dataset, seed=seed, anomaly_class_ind=ac, method=method);
#orig_data = GenerativeAD.Datasets.normalize_data(orig_data);
#orig_data_denormalized = GenerativeAD.load_data(dataset, seed=seed, anomaly_class_ind=ac, method=method, 
#    denormalize=true);



if modelname in ["sgvae", "cgn"]
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

# this will have to be specific for each modelname
#if "weights" in readdir(mf)
if modelname == "sgvae"
    model = GenerativeAD.Models.SGVAE(load_sgvae_model(mf, device))
else
    error("unknown modelname $modelname")
end

# load the original experiment file
expf = filter(x->occursin("$(model_id)", x), expfs)
expf = filter(x->!occursin("model", x), expf)
# put a return here in case this is empty
expf = joinpath(exppath, expf[1])
exptime = mtime(expf)
expdata = load(expf)

# first do inference on the original data
tr_scores, val_scores_orig, tst_scores_orig = map(x->StatsBase.predict(model, x), (orig_data[1][1],
    orig_data[2][1][:,:,:,orig_data[2][2] .== 0], orig_data[3][1][:,:,:,orig_data[3][2] .== 0]))

# also load the new data for inference
if dataset == "wildlife_MNIST"

else
    error("unkown dataset $(dataset)")
end


if exptime > norm_time


"""
x = orig_data[1][1][:,:,:,1:10];
x = orig_data_denormalized[1][1][:,:,:,1:10];
y = orig_data[1][2][1:10];
StatsBase.predict(model, x)
expdata[:tr_scores][1:10]

"""