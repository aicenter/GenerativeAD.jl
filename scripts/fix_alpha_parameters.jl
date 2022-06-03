using DrWatson
@quickactivate
using FileIO,  BSON, DataFrames
using GenerativeAD
using ArgParse
using ProgressMeter

s = ArgParseSettings()
@add_arg_table! s begin
    "source"
		arg_type = String
		default = "sgad_alpha_evaluation_kp/images_leave-one-in"
		help = "Data prefix of generated files."
end
parsed_args = parse_args(ARGS, s)
@unpack source = parsed_args

source = datadir(source)
files = GenerativeAD.Evaluation.collect_files_th(source)

function fix_alpha_parameters(f)
	fname = basename(f)
	beta = get(parse_savename("_"*fname)[2] , "beta", NaN)
	d = load(f)[:df]
	parameters = d[:parameters][1]
	parameters = parse_savename("_"*parameters)[2]
	parameters = NamedTuple{Tuple(Symbol.(keys(parameters)))}(values(parameters))
	parameters = !isnan(beta) ? merge(parameters, (beta=beta,)) : parameters
	d[:parameters][1] = parameters
	save(f, :df => d)
end

@showprogress for f in files
	fix_alpha_parameters(f)
end
