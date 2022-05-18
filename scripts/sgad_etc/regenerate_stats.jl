using ArgParse
using DrWatson
@quickactivate
using BSON
using Random
using FileIO
using DataFrames
using Base.Threads: @threads
using GenerativeAD
using GenerativeAD.Evaluation: _prefix_symbol, _get_anomaly_class, _auc_at_subsamples_anomalous
using GenerativeAD.Evaluation: BASE_METRICS, AUC_METRICS, compute_stats
using StatsBase, Random
using ProgressMeter

# pkgs which come from deserialized BSONs
# have to be present in the Main module
using ValueHistories
using LinearAlgebra
using EvalMetrics

s = ArgParseSettings()
@add_arg_table! s begin
    "project_dir"
		arg_type = String
		default = "/home/skvarvit/generativead/GenerativeAD.jl"
		help = "Data prefix of experiment files."
	"model"
		arg_type = String
		default = "vae"
		help = "Which model to process"
	"dataset"
		arg_type = String
		default = "CIFAR10"
		help = "Which dataset to process"
	"-f", "--force"
    	action = :store_true
		help = "Overwrite all generated files."
end
parsed_args = parse_args(ARGS, s)
@unpack project_dir, model, dataset, force = parsed_args
ignore_higher = model != "cgn"

"""
	generate_stats(source_dir::String, model::String, dataset::String; force=true)

Collects all the results from experiments in datadir prefix `source_prefix`, 
computes evaluation metrics and stores results in datadir prefix `target_prefix` 
while retaining the folder structure. If `force=true` the function overwrites 
already precomputed results. 
"""
function generate_stats(project_dir::String, model::String, dataset::String; force=true, ignore_higher=true)
	datatype = (dataset in ["CIFAR10", "SVHN2", "wildlife_MNIST"]) ? "leave-one-in" : "mvtec"
	source_dir = joinpath(abspath(project_dir), "data/experiments/images_$(datatype)/$(model)/$(dataset)")
	out_dir = datadir("evaluation_kp/images_$(datatype)/$(model)/$(dataset)")

	@info "Collecting files from $(source_dir) folder."
	files = GenerativeAD.Evaluation.collect_files_th(source_dir; ignore_higher=ignore_higher);
	# filter out model files
	filter!(x -> !startswith(basename(x), "model"), files);
	filter!(x -> !occursin(".pth", basename(x)), files);

	@info "Collected $(length(files)) files from $(source_dir) folder."
	# it might happen that when appending results some of the cores just go over already computed files
	files = files[randperm(length(files))]
	p = Progress(length(files))

	@threads for f in files
		target = joinpath(out_dir, split(dirname(f), source_dir)[2][2:end], "eval_$(basename(f))")
		try
			if (isfile(target) && force) || ~isfile(target)
				r = load(f)
				df = compute_stats(r; top_metrics=false, top_metrics_new=true)
				wsave(target, Dict(:df => df))
				#@info "Saving evaluation results at $(target)"
			end
		catch e
			# remove old files in order to ensure consistency
			if (isfile(target) && force)
				rm(target)
			end
			@warn "Processing of $f failed due to \n$e"
		end
	    next!(p)
	end
end

generate_stats(project_dir, model, dataset; force=force, ignore_higher=ignore_higher)
@info "---------------- DONE -----------------"
