using ArgParse
using DrWatson
@quickactivate
using BSON
using Random
using FileIO
using DataFrames
using Base.Threads: @threads
using GenerativeAD

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
	files = GenerativeAD.Evaluation.collect_files(source)
	@info "Collected $(length(files)) files from $source folder."

	frames = Vector{DataFrame}(undef, length(files))
	@threads for i in 1:length(files)
		df = load(files[i])[:df]
		frames[i] = df
	end
	vcat(frames...)
end


function main(args)
	df = collect_stats(args["source_prefix"])
	f = datadir(args["filename"])
	if (isfile(f) && args["force"]) || ~isfile(f)
		@info "Saving $(nrow(df)) rows to $f."
		wsave(f, Dict(:df => df))
	end
	@info "---------------- DONE -----------------"
end

main(parse_args(ARGS, s))
