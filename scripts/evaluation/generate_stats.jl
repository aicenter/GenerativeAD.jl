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
		default = "experiments/tabular"
		help = "Data prefix of experiment files."
	"target_prefix"
		arg_type = String
		default = "evaluation/tabular"
		help = "Data prefix of generated files."
	"-f", "--force"
    	action = :store_true
		help = "Overwrite all generated files."
end

"""
	generate_stats(source_prefix::String, target_prefix::String; force=true)

Collects all the results from experiments in datadir prefix `source_prefix`, 
computes evaluation metrics and stores results in datadir prefix `target_prefix` 
while retaining the folder structure. If `force=true` the function overwrites 
already precomputed results. 
"""
function generate_stats(source_prefix::String, target_prefix::String; force=true)
	(source_prefix == target_prefix) && error("Results have to be stored in different folder.")
	
	source = datadir(source_prefix)
	@info "Collecting files from $source folder."
	files = GenerativeAD.Evaluation.collect_files(source)
	# filter out model files
	filter!(x -> !startswith(basename(x), "model"), files)
	@info "Collected $(length(files)) files from $source folder."
	# it might happen that when appending results some of the cores
	# just go over already computed files
	files = files[randperm(length(files))]

	@threads for f in files
		try
			target_dir = dirname(replace(f, source_prefix => target_prefix))
			target = joinpath(target_dir, "eval_$(basename(f))")
			if (isfile(target) && force) || ~isfile(target)
				df = GenerativeAD.Evaluation.compute_stats(f)
				wsave(target, Dict(:df => df))
				@info "Saving evaluation results at $(target)"
			end
		catch e
			@warn "Processing of $f failed due to \n$e"
		end
	end
end

function main(args)
	generate_stats(
		args["source_prefix"], 
		args["target_prefix"]; 
		force=args["force"])
	@info "---------------- DONE -----------------"
end

main(parse_args(ARGS, s))