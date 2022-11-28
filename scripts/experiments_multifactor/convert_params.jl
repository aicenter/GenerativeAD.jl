# this converts params saved in eval files from named tuples to strings, which is much easier to work with
# also, we only need to do it on the alpha files
using DrWatson
@quickactivate
using FileIO, BSON, DataFrames
using GenerativeAD
using ProgressMeter
using ArgParse

s = ArgParseSettings()
@add_arg_table! s begin
   "modelname"
        default = "sgvaegan100"
        arg_type = String
        help = "modelname"
    "dataset"
        default = "wildlife_MNIST"
        arg_type = String
        help = "dataset or mvtec category"
end
parsed_args = parse_args(ARGS, s)
@unpack modelname, dataset = parsed_args


target = datadir("experiments_multifactor/alpha_evaluation_mf_normal/$(modelname)/$(dataset)")
@info "Collecting files from $target..."
evalfs = GenerativeAD.Evaluation.collect_files(target)

function convert_params(ef)
	d = load(ef)[:df]
	ps = d.parameters[1]
	if !(typeof(ps) <: NamedTuple)
		return
	end
	ps = savename(ps)
	d.parameters[1] = ps
	save(ef, :df => d)
	ps
end

@info "Processing $(length(evalfs)) files..."
@showprogress map(evalfs) do x
	convert_params(x)
end