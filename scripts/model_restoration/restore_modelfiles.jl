# the purpose of this script is to fix the wrongly saved value of zdim model files
using DrWatson
@quickactivate
using BSON, FileIO
using Flux
using GenerativeModels
using DistributionsAD
using ValueHistories
using GenerativeAD
using ArgParse
using ProgressMeter

s = ArgParseSettings()
@add_arg_table! s begin
	"path"
		arg_type = String
		help = "path"
end
parsed_args = parse_args(ARGS, s)
@unpack path = parsed_args

files = GenerativeAD.Evaluation.collect_files(path)
mfiles = filter(f->occursin("model", f), files)
@info "Restoring modelfiles in $path"

function fix_modelfile(mf) 
	# get parameters
	savepath = dirname(mf)
	parameters = parse_savename(mf)[2]
	parameters = NamedTuple{Tuple(Symbol.(keys(parameters)))}(values(parameters))

	# get additional fit info
	# zdim is empty since there was a bug that saved it incorrectly in model file
	infopars = merge(parameters, (score="latent", zdim="",))
	try
		infof = filter(x->occursin(savename(infopars), x), files)[1]
		global info = load(infof)
	catch e
		@info "data for $mf not found"
		return ""
	end

	# now create the new parameters
	outpars = merge(parameters, (zdim=info[:parameters].zdim,))

	# now save the fixed model data and delete the old model
	sn = joinpath(savepath, savename("model", outpars, "bson"))
	if sn != mf # only do all of this if the old and new modelfiles are different
		# get model data
		model_data = load(mf)
		# also add the additional fields
		model_data["history"] = info[:history]
		model_data["fit_t"] = info[:fit_t]
		model_data["parameters"] = info[:parameters]

		save(sn, model_data)
		rm(mf)
	end

	return sn
end

@showprogress map(fix_modelfile, mfiles)
