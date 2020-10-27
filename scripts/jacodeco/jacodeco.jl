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
	"datatype"
		default = "tabular"
		arg_type = String
		help = "tabular or image"
	"dataset"
		default = "iris"
		arg_type = String
		help = "dataset"
	
end
parsed_args = parse_args(ARGS, s)
@unpack dataset, datatype, modelname = parsed_args

masterpath = datadir("experiments/$(datatype)/$(modelname)/$(dataset)")
files = GenerativeAD.Evaluation.collect_files(masterpath)
mfiles = filter(f->occursin("model", f), files)

function save_jacodeco(f::String, data, seed::Int)
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
	result = (x -> -GenerativeAD.Models.jacodeco(model, x), 
		merge(mdata["parameters"], (score = "jacodeco",)))

	GenerativeAD.experiment(result..., data, savepath; save_entries...)
end

for f in mfiles
	# get data
	savepath = dirname(f)
	seed = parse(Int, replace(basename(savepath), "seed=" => ""))
	data = GenerativeAD.load_data(dataset, seed=seed)
	
	# compute and save the score
	@info "computing jacodeco for $f"
	save_jacodeco(f, data, seed)
end
