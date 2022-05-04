using ArgParse
using DrWatson
@quickactivate
using BSON
using Random
using FileIO
using DataFrames
using GenerativeAD
using ProgressMeter

s = ArgParseSettings()
@add_arg_table! s begin
	"chunk_index"
		arg_type = Int
		help = "index of chunk"
    "source_prefix"
		arg_type = String
		default = "evaluation/images"
		help = "Data prefix of generated files."
	"out_dir"
		arg_type = String
		default = "evaluation/images_eval_cache"
		help = "Where to store the cached DataFrames."
	"chunk_size"
		arg_type = Int
		default = 10000
		help = "how many results to be concatenated in a single chunk"
	"-f", "--force"
    	action = :store_true
		help = "Overwrite all generated files."
end

"""
	collect_stats(source_prefix::String)

Collects evaluation DataFrames from a given data folder prefix `source_prefix`.
"""
function collect_stats(part)
	fs = []
	@showprogress for f in part
		push!(fs, load(f)[:df])
	end
	reduce(vcat, fs)
end

function main(args)
	# get argvals
	source_prefix = args["source_prefix"]
	chunk_index = args["chunk_index"]
	out_dir = args["out_dir"]
	chunk_size = args["chunk_size"]
	force = args["force"]


	source = datadir(source_prefix)
	@info "Collecting files from $source folder."
	files = GenerativeAD.Evaluation.collect_files_th(source)
	@info "Collected $(length(files)) files from $source folder."
	parts = collect(Iterators.partition(files, chunk_size))
	nparts = length(parts)
	if chunk_index > nparts
		@info "Requested chunk index $(chunk_index) is out of range."
		return nothing
	end
	out_dir = datadir(out_dir)
	mkpath(out_dir)
	f = joinpath(out_dir, "$(chunk_index)-$(nchunks).bson")
	if (isfile(f) && args["force"]) || ~isfile(f)
		df = collect_stats(parts[chunk_index])
		@info "Loaded partition with index $(chunk_index)."
		@info "Saving $(nrow(df)) rows to $f."
		wsave(f, Dict(:df => df))
	else
		@info "skipping computation of $f"
	end
	@info "---------------- DONE -----------------"
end

main(parse_args(ARGS, s))
