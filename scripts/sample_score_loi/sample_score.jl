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
		default = "vae"
		arg_type = String
		help = "model name"
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
@unpack dataset, modelname, seed, anomaly_class = parsed_args

masterpath = datadir("experiments/images_leave-one-in/$(modelname)/$(dataset)")
files = GenerativeAD.Evaluation.collect_files(masterpath)
mfiles = filter(f->occursin("model", f), files)
if seed != nothing
	filter!(x->occursin("/seed=$seed/", x), mfiles)
end
if anomaly_class != nothing
	filter!(x->occursin("/ac=$(anomaly_class)/", x), mfiles)
end

sample_score_batched(m,x,L,batchsize) = 
	vcat(map(y-> Base.invokelatest(GenerativeAD.Models.reconstruction_score, m, y, L), 
		Flux.Data.DataLoader(x, batchsize=batchsize))...)
sample_score_batched_gpu(m,x,L,batchsize) = 
	vcat(map(y-> cpu(Base.invokelatest(GenerativeAD.Models.reconstruction_score, m, 
		gpu(Array(y)), L)), Flux.Data.DataLoader(x, batchsize=batchsize))...)

function save_sample_score(f::String, data, seed::Int, ac=nothing)
	# get model
	savepath = dirname(f)
	mdata = load(f)
	model = mdata["model"]
	L = 100

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
	save_entries = (ac == nothing) ? save_entries : merge(save_entries, (anomaly_class=ac,))
	result = (x -> sample_score_batched_gpu(gpu(model), x, L, 256), 
		merge(mdata["parameters"], (L = L, score = "reconstruction-sampled"))) 

	# if the file does not exist already, compute the scores
	savef = joinpath(savepath, savename(result[2], "bson", digits=5))
	if !isfile(savef)
		@info "computing sample score for $f"
		GenerativeAD.experiment(result..., data, savepath; save_entries...)
	end
end

for f in mfiles
	# get data
	savepath = dirname(f)
	local seed = parse(Int, replace(basename(savepath), "seed=" => ""))
	ac = occursin("ac=", savepath) ? parse(Int, replace(basename(dirname(savepath)), "ac=" => "")) : nothing
	data = GenerativeAD.load_data(dataset, seed=seed, anomaly_class_ind=ac, 
		method="leave-one-in")
		
	# compute and save the score
	save_sample_score(f, data, seed, ac)
end
@info "DONE"
