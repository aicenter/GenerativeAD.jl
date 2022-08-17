using ArgParse
using DrWatson
@quickactivate
using BSON
using Random
using FileIO
using DataFrames
using Base.Threads: @threads
using GenerativeAD

const CHUNK_SIZE = 1000

s = ArgParseSettings()
@add_arg_table! s begin
    "source_prefix"
		arg_type = String
		default = "evaluation/images"
		help = "Data prefix of generated files."
	"filename"
		arg_type = String
		default = "evaluation/images_eval.bson"
		help = "Where to store the cached DataFrame."
	"--postprocess"
		action = :store_true
		help = "convert tuple parameters to strings for faster loading times"
	"-f", "--force"
    	action = :store_true
		help = "Overwrite all generated files."
end

"""
	collect_stats(source_prefix::String)

Collects evaluation DataFrames from a given data folder prefix `source_prefix`.
"""
function collect_stats(source_prefix::String)
	source = datadir(source_prefix)
	@info "Collecting files from $source folder."
	files = GenerativeAD.Evaluation.collect_files_th(source)
	@info "Collected $(length(files)) files from $source folder."

	parts = collect(Iterators.partition(files, CHUNK_SIZE))
	frames = Vector{DataFrame}(undef, length(parts))
	@info "Loading into $(length(parts)) partitions."
	@threads for i in 1:length(parts)
		df = reduce(vcat, [load(f)[:df] for f in parts[i]])
		frames[i] = df
		print("-")
	end
	@info "Loaded $(length(parts)) partitions."
	df = reduce(vcat, frames)
end

# this is mainyl used for alpha evaluation results
function postprocess(df)
	parameters = df.parameters
	df.parameters = savename.(parameters)
	df.weights_texture = [get(p, :weights_texture, NaN) for p in parameters]
	df.init_alpha = [get(p, :init_alpha, NaN) for p in parameters]
	df.alpha0 = [get(p, :alpha0, NaN) for p in parameters]

	# delete nan columns
	for col in names(df)
		if occursin("auc", col) && all(isnan.(df[col]))
			df = df[:,[p for p in names(df) if p != col]]
		end
	end
	df
end

function main(args)
	df = collect_stats(args["source_prefix"])
	df = args["postprocess"] ? postprocess(df) : df
	f = datadir(args["filename"])
	if (isfile(f) && args["force"]) || ~isfile(f)
		@info "Saving $(nrow(df)) rows to $f."
		wsave(f, Dict(:df => df))
	end
	@info "---------------- DONE -----------------"
end

main(parse_args(ARGS, s))
