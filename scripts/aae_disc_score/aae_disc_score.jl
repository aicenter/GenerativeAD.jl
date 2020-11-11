using DrWatson
@quickactivate
using ArgParse
using GenerativeAD
import StatsBase: fit!, predict
using StatsBase
using BSON, FileIO
using Flux
using GenerativeModels
using DistributionsAD
using ValueHistories

s = ArgParseSettings()
@add_arg_table! s begin
	"modelname"
		default = "aae"
		arg_type = String
		help = "model name"
	"datatype"
		default = "tabular"
		arg_type = String
		help = "tabular or image"
	"dataset"
		default = "iris"
		arg_type = String
		help = "dataset"
	"--seed"
		default = nothing
		help = "if specified, only results for a given seed will be recomputed"
	"--anomaly_class"
		default = nothing
		help = "if specified, only results for a given anomaly class will be recomputed"
end
parsed_args = parse_args(ARGS, s)
@unpack dataset, datatype, modelname, seed, anomaly_class = parsed_args

masterpath = datadir("experiments/$(datatype)/$(modelname)/$(dataset)")
files = GenerativeAD.Evaluation.collect_files(masterpath)
mfiles = filter(f->occursin("model", f), files)
if seed != nothing
	filter!(x->occursin("/seed=$seed/", x), mfiles)
end
if anomaly_class != nothing
	filter!(x->occursin("/ac=$(anomaly_class)/", x), mfiles)
end

aae_score_batched(m,x,alpha,batchsize) = 
	vcat(map(y-> Base.invokelatest(GenerativeAD.Models.aae_score, m, y, alpha), Flux.Data.DataLoader(x, batchsize=batchsize))...)
aae_score_batched_gpu(m,x,alpha,batchsize) = 
	vcat(map(y-> cpu(Base.invokelatest(GenerativeAD.Models.aae_score, m, gpu(Array(y)), alpha)), Flux.Data.DataLoader(x, batchsize=batchsize))...)

function save_aae_disc_score(f::String, data, seed::Int, ac=nothing)
	# get model
	savepath = dirname(f)
	mdata = load(f)
	model = mdata["model"]

	# setup entries to be saved
	save_entries = (
		modelname = modelname,
		fit_t = mdata["fit_t"],
		history = mdata["history"],
		dataset = dataset,
		npars = sum(map(p->length(p), Flux.params(model))),
		model = nothing,
		seed = seed
		)
	save_entries = (ac == nothing) ? save_entries : merge(save_entries, (ac=ac,))
	if ac == nothing
		results = [(x -> aae_score_batched(model, x, alpha, 512), 
			merge(mdata["parameters"], (alpha = alpha, score = "disc"))) 
			for alpha in 0f0:0.1f0:1f0]
	else
		results = [(x -> aae_score_batched_gpu(gpu(model), x, alpha, 512), 
			merge(mdata["parameters"], (alpha = alpha, score = "disc"))) 
			for alpha in 0f0:0.1f0:1f0]
	end

	# if the file does not exist already, compute the scores
	for result in results
		savef = joinpath(savepath, savename(result[2], "bson", digits=5))
		if !isfile(savef)
			@info "computing AAE discriminator score for $f"
			GenerativeAD.experiment(result..., data, savepath; save_entries...)
		end
	end
end

for f in mfiles
	# get data
	savepath = dirname(f)
	local seed = parse(Int, replace(basename(savepath), "seed=" => ""))
	ac = occursin("ac=", savepath) ? parse(Int, replace(basename(dirname(savepath)), "ac=" => "")) : nothing
	data = (ac == nothing) ? 
		GenerativeAD.load_data(dataset, seed=seed) : 
		GenerativeAD.load_data(dataset, seed=seed, anomaly_class_ind=ac)
		
	# compute and save the score
	save_jacodeco(f, data, seed, ac)
end
