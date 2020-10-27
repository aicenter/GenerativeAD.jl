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
	"--force"
		action = :store_true
		help = "force overwriting"
end
parsed_args = parse_args(ARGS, s)
@unpack path, force = parsed_args

path = abspath(path)
files = GenerativeAD.Evaluation.collect_files(path)
mfiles = filter(f->occursin("model", f), files)
@info "storing modelfiles in $path"

function fix_modelfile(mf) 
	# get parameters
	savepath = dirname(mf)
	parameters = parse_savename(mf)[2]
	parameters = NamedTuple{Tuple(Symbol.(keys(parameters)))}(values(parameters))

	# get additional fit info
	# zdim is empty since there was a bug that saved it incorrectly in model file
	#infopars = merge(parameters, (score="latent", zdim="",lr=""))
	try
		infof = filter(x->occursin("$(parameters.init_seed)", x), files)[1]
		global info = load(infof)
	catch e
		#@info "data for $mf not found"
		return ""
	end

	# now create the new parameters
	outpars = merge(parameters, (zdim=info[:parameters].zdim,lr=info[:parameters].lr))

	# now save the fixed model data and delete the old model
	sn = joinpath(savepath, savename("model", outpars, "bson",digits=5))
	if ((sn != mf) || force) # only do all of this if the old and new modelfiles are different
		# get model data
		model_data = load(mf)
		# also add the additional fields
		model_data["history"] = info[:history]
		model_data["fit_t"] = info[:fit_t]
		model_data["parameters"] = Base.structdiff(info[:parameters], (score=nothing,))

		rm(mf)
		save(sn, model_data)
	end

	return sn
end

newfiles = @showprogress map(fix_modelfile, mfiles)

# print files that were not modified
@info "these files were not updated, probably missing the corresponding score files:"
for f in mfiles[newfiles .== ""]
	println(f)
end
