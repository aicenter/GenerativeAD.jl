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
using GenerativeAD.Evaluation: BASE_METRICS, AUC_METRICS
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
				df = compute_stats(r)
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

function compute_stats(r::Dict{Symbol,Any})
	row = (
		modelname = r[:modelname],
		dataset = Symbol("dataset") in keys(r) ? r[:dataset] : "MVTec-AD_" * r[:category], # until the files are fixed
		phash = hash(r[:parameters]),
		parameters = savename(r[:parameters], digits=6), 
		fit_t = r[:fit_t],
		tr_eval_t = r[:tr_eval_t],
		tst_eval_t = r[:tst_eval_t],
		val_eval_t = r[:val_eval_t],
		seed = r[:seed],
		npars = (Symbol("npars") in keys(r)) ? r[:npars] : 0
	)
	
	max_seed = 10	
	anomaly_class = _get_anomaly_class(r)
	if anomaly_class != -1
		row = merge(row, (anomaly_class = anomaly_class,))
	end

	# add fs = first stage fit/eval time
	# case of ensembles and 2stage models
	if Symbol("encoder_fit_t") in keys(r)
		row = merge(row, (fs_fit_t = r[:encoder_fit_t], fs_eval_t = r[:encode_t],))
	elseif Symbol("ensemble_fit_t") in keys(r)
		row = merge(row, (fs_fit_t = r[:ensemble_fit_t], fs_eval_t = r[:ensemble_eval_t],))
	else
		row = merge(row, (fs_fit_t = 0.0, fs_eval_t = 0.0,))
	end

	for splt in ["val", "tst"]
		scores = r[_prefix_symbol(splt, :scores)]
		labels = r[_prefix_symbol(splt, :labels)]

		if length(scores) > 1
			# in cases where scores is not an 1D array
			scores = scores[:]

			invalid = isnan.(scores)
			ninvalid = sum(invalid)

			if ninvalid > 0
				invrat = ninvalid/length(scores)
				invlab = labels[invalid]
				cml = countmap(invlab)
				# we have lost the filename here due to the interface change
				@warn "Invalid stats for $(r[:modelname])/$(r[:dataset])/.../$(row[:parameters]) \t $(ninvalid) | $(invrat) | $(length(scores)) | $(get(cml, 1.0, 0)) | $(get(cml, 0.0, 0))"

				scores = scores[.~invalid]
				labels = labels[.~invalid]
				(invrat > 0.5) && error("$(splt)_scores contain too many NaN")
			end

			roc = EvalMetrics.roccurve(labels, scores)
			auc = EvalMetrics.auc_trapezoidal(roc...)
			prc = EvalMetrics.prcurve(labels, scores)
			auprc = EvalMetrics.auc_trapezoidal(prc...)

			t5 = EvalMetrics.threshold_at_fpr(labels, scores, 0.05)
			cm5 = ConfusionMatrix(labels, scores, t5)
			tpr5 = EvalMetrics.true_positive_rate(cm5)
			f5 = EvalMetrics.f1_score(cm5)

			row = merge(row, (;zip(_prefix_symbol.(splt, 
					BASE_METRICS), 
					[auc, auprc, tpr5, f5])...))

			# compute auc on a randomly selected portion of samples
			if splt == "val"
				auc_ano_100 = [mean([_auc_at_subsamples_anomalous(p/100, 1.0, labels, scores, seed=s) for s in 1:max_seed]) 
					for p in [100.0, 50.0, 20.0, 10.0, 5.0, 2.0, 1.0, 0.5, 0.2, 0.1]]
				row = merge(row, (;zip(_prefix_symbol.(splt, map(x->x * "_100", AUC_METRICS)), auc_ano_100)...))

				auc_ano_50 = [mean([_auc_at_subsamples_anomalous(p/100, 0.5, labels, scores, seed=s) for s in 1:max_seed]) 
					for p in [100.0, 50.0, 20.0, 10.0, 5.0, 2.0, 1.0, 0.5, 0.2, 0.1]]
				row = merge(row, (;zip(_prefix_symbol.(splt, map(x->x * "_50", AUC_METRICS)), auc_ano_50)...))

				auc_ano_10 = [mean([_auc_at_subsamples_anomalous(p/100, 0.1, labels, scores, seed=s) for s in 1:max_seed]) 
					for p in [100.0, 50.0, 20.0, 10.0, 5.0, 2.0, 1.0, 0.5, 0.2, 0.1]]
				row = merge(row, (;zip(_prefix_symbol.(splt, map(x->x * "_10", AUC_METRICS)), auc_ano_10)...))

				prop_ps = [100, 50, 20, 10, 5, 2, 1]
				auc_prop_100 = [mean([_auc_at_subsamples_anomalous(1.0, p/100, labels, scores, seed=s) for s in 1:max_seed]) 
					for p in prop_ps]
				row = merge(row, (;zip(_prefix_symbol.(splt, map(x-> "auc_100_$(x)", prop_ps)), auc_prop_100)...))
			end
		else
			error("$(splt)_scores contain only one value")
		end
	end

	DataFrame([row])
end

generate_stats(project_dir, model, dataset; force=force, ignore_higher=ignore_higher)
@info "---------------- DONE -----------------"
