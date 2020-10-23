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
	"dataset"
		default = "iris"
		arg_type = String
		help = "dataset"
	"modelname"
		default = "vae"
		arg_type = String
		help = "model name"
end
parsed_args = parse_args(ARGS, s)
@unpack dataset, modelname = parsed_args

masterpath = datadir("experiments/tabular/$(modelname)/$(dataset)")
files = GenerativeAD.Evaluation.collect_files(masterpath)
mfiles = filter(f->occursin("model", f), files)

function save_jacodeco(f::String, data, seed::Int)
	# get model
	savepath = dirname(f)
	parameters = parse_savename(f)[2]
	parameters = NamedTuple{Tuple(Symbol.(keys(parameters)))}(values(parameters))
	model = load(f)["model"]

	# get additional fit info
	# zdim is empty since there was a bug that saved it incorrectly in model file
	infopars = merge(parameters, (score="latent", zdim="",))
	#try
	infof = filter(x->occursin(savename(infopars), x), files)[1]
	info = load(infof)
	fit_t = info[:fit_t]
	history = info[:history]
	zdim = info[:parameters].zdim
    #catch e
	#	@info "No scores saved with $f found or loaded."
	#	fit_t = NaN
	#	history = nothing
	#	zdim = parameters.zdim # this is probably wrong and might make trouble later
	#end 

	# some of this stuff is only recoverable from the score files but they are quite difficult 
	# to load automatically
	save_entries = (
		modelname = modelname,
		fit_t = fit_t,
		history = history,
		dataset = dataset,
		npars = sum(map(p->length(p), Flux.params(model))),
		model = nothing,
		seed = seed
		)
	result = (x -> -GenerativeAD.Models.jacodeco(model, x), 
		merge(parameters, (score = "jacodeco",zdim = zdim)))

	GenerativeAD.experiment(result..., data, savepath; save_entries...)
end

for f in mfiles
	# get data
	savepath = dirname(f)
	seed = parse(Int, replace(basename(savepath), "seed=" => ""))
	data = GenerativeAD.load_data(dataset, seed=seed)

	# compute and save the score
	save_jacodeco(f, data, seed)
end
