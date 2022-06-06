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
	beta = occursin("beta", fname) ? get(parse_savename("beta"*split(fname, "beta")[2])[2] , "beta", NaN) : NaN
	try
		d = load(f)[:df]
		parameters = d[:parameters][1]
		if (typeof(parameters) <: NamedTuple)
			if haskey(parameters, :beta)
				return
			end
		else
			parameters = parse_savename("_"*parameters)[2]
			parameters = NamedTuple{Tuple(Symbol.(keys(parameters)))}(values(parameters))
		end
		parameters = !isnan(beta) ? merge(parameters, (beta=beta,)) : parameters
		d[:parameters][1] = parameters
		save(f, :df => d)
	catch e
		@info "$(typeof(e)) error when processing $f"
	end
end

@showprogress for f in files
	fix_alpha_parameters(f)
end
