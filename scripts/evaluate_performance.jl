using DrWatson
@quickactivate
using EvalMetrics
using FileIO
using BSON
using DataFrames
using PrettyTables
using PrettyTables.Crayons
using Statistics
using StatsBase
using Base.Threads: @threads

# pkgs which come from BSONs
using ValueHistories
using LinearAlgebra

# metric names and settings 
const BASE_METRICS = ["auc", "auprc", "tpr_5", "f1_5"]
const PAT_METRICS = ["pat_1", "pat_5", "pat_10", "pat_20"]

"""
	_prefix_symbol(prefix, s)

Modifies symbol `s` by adding `prefix` with underscore.
"""
_prefix_symbol(prefix, s) = Symbol("$(prefix)_$(s)")


"""
	_precision_at(p, scores, labels)

Computes precision on portion `p` samples with highest score.
"""
function _precision_at(p, labels, scores)
	pN = Int(round(floor(p*length(labels)))) # round is more forgiving
	if pN > 0
		sp = sortperm(scores, rev=true)[1:pN]
		# @info sp scores[sp] labels[sp]
		return EvalMetrics.precision(labels[sp], ones(eltype(labels), pN))
	else
		return NaN
	end
end


"""
	compute_stats(f::String)

Computes evaluation metrics from the results of experiment in serialized bson at path `f`.
Returns a DataFrame row with metrics and additional metadata for groupby's.
Hash of the model parameters is precomputed in order to make the groupby easier.
"""
function compute_stats(f::String)
	r = load(f)
	row = (
		modelname = r[:modelname],
		dataset = r[:dataset],
		phash = hash(r[:parameters]),
		parameters = savename(r[:parameters], digits=6), 
		seed = r[:seed])
	
	if Symbol("anomaly_class") in keys(r)
		row = merge(row, (anomaly_class = r[:anomaly_class],))
	end

	for splt in ["val", "tst"]
		scores = r[_prefix_symbol(splt, :scores)]
		labels = r[_prefix_symbol(splt, :labels)]

		if length(scores) > 1
			invalid = isnan.(scores)
			ninvalid = sum(invalid)

			if ninvalid > 0
				invrat = ninvalid/length(scores)
				invlab = labels[invalid]
				cml = countmap(invlab)
				@warn "Invalid stats for $(f) \t $(ninvalid) | $(invrat) | $(length(scores)) | $(get(cml, 1.0, 0)) | $(get(cml, 0.0, 0))"

				scores = scores[.~invalid]
				labels = labels[.~invalid]
				(invrat > 0.5) && error("$(splt)_scores contain too many NaN")
			end
			
			# in cases where the score is not 1D array
			scores = scores[:]

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

			# compute precision on most anomalous samples
			pat = [_precision_at(p/100, labels, scores) for p in [1, 5, 10, 20]]
			row = merge(row, (;zip(_prefix_symbol.(splt, PAT_METRICS), pat)...))
		else
			error("$(splt)_scores contain only one value")
		end
	end

	DataFrame([row])
end

function collect_files!(target::String, files)
	if isfile(target)
		push!(files, target)
	else
		for file in readdir(target, join=true)
			collect_files!(file, files)
		end
	end
	files
end


"""
	collect_files(target)

Walks recursively the `target` directory, collecting all files only along the way.
"""
collect_files(target) = collect_files!(target, [])


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
	files = collect_files(source)
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
				df = compute_stats(f)
				wsave(target, Dict(:df => df))
				@info "Saving evaluation results at $(target)"
			end
		catch e
			@warn "Processing of $f failed due to \n$e"
		end
	end
end


"""
	collect_stats(source_prefix::String)

Collects evaluation DataFrames from a given data folder prefix `source_prefix`.
"""
function collect_stats(source_prefix::String)
	source = datadir(source_prefix)
	@info "Collecting files from $source folder."
	files = collect_files(source)
	@info "Collected $(length(files)) files from $source folder."

	frames = Vector{DataFrame}(undef, length(files))
	@threads for i in 1:length(files)
		df = load(files[i])[:df]
		frames[i] = df
	end
	vcat(frames...)
end

"""
	aggregate_stats(df::DataFrame, criterion_col=:val_auc, output_cols=[:tst_auc])

Chooses agregates eval metrics by seed over a given hyperparameter and then chooses best
model based on `criterion_col`. By default the output contains `dataset`, `modelname` and
`samples` columns, where the last represents the number of hyperparameter combinations
over which the maximum is computed. Addional comlumns can be specified using `output_cols`.
"""
function aggregate_stats(df::DataFrame, criterion_col=:val_auc, output_cols=[:tst_auc, :tst_auc_std]; verbose=false)
	agg_cols = vcat(_prefix_symbol.("val", BASE_METRICS), _prefix_symbol.("tst", BASE_METRICS))
	agg_cols = vcat(agg_cols, _prefix_symbol.("val", PAT_METRICS), _prefix_symbol.("tst", PAT_METRICS))

	# agregate by seed over given hyperparameter and then choose best
	results = []
	for (dkey, dg) in pairs(groupby(df, :dataset))
		if verbose
			@info "Processing $(dkey.dataset) with $(nrow(dg)) trained models."
		end
		for (mkey, mg) in pairs(groupby(dg, :modelname))
			pg = groupby(mg, :phash)
			pg_agg = combine(pg, :parameters => unique => :parameters, agg_cols .=> std, agg_cols .=> mean .=> agg_cols)
			if verbose
				@info "\t$(mkey.modelname) with $(nrow(pg_agg)) experiments."
			end
			best = first(sort!(pg_agg, order(criterion_col, rev=true)))
			row = merge(
				(dataset = dkey.dataset, modelname = mkey.modelname, samples=nrow(pg_agg)), 
				(;zip(output_cols, [best[oc] for oc in output_cols])...))
			push!(results, DataFrame([row]))
		end
	end
	vcat(results...)
end

"""
	print_table(df::DataFrame)

Prints dataframe with columns [:modelname, :dataset] and one scalar variable column
given by `metric_col` argument. By default highlights maximum value in each row.
"""
function print_table(df::DataFrame, metric_col=:tst_auc)
	# check if column names are present
	(!(String(metric_col) in names(df)) || !("modelname" in names(df)) || !("dataset" in names(df))) && error("Incorrect column names.")

	# get all the models that are present in the dataframe
	all_models = unique(df.modelname)

	# transposing the dataframe
	results = []
	for (dkey, dg) in pairs(groupby(df, :dataset))
		models = dg[:modelname]
		metric = dg[metric_col]
		row = merge(
			(dataset = dkey.dataset,),
			(;zip(Symbol.(models), metric)...))
		
		# when there are no evaluation files for some models on particular dataset
		if nrow(dg) < length(all_models)
			mm = setdiff(all_models, models)
			row = merge(row, (;zip(Symbol.(mm), fill(NaN, length(mm)))...))
		end
		push!(results, DataFrame([row]))
	end

	# pretty print
	ultimate = vcat(results...)
	sort!(ultimate, :dataset)

	# average rank
	mask_nan_max = (x) -> (isnan(x) ? -Inf : x)
	rs = zeros(size(ultimate, 2) - 1)
	for row in eachrow(ultimate)
		rs .+= StatsBase.competerank(mask_nan_max.(Vector(row[2:end])), rev = true)
	end
	rs ./= size(ultimate, 1)
	push!(ultimate, ["--- RANK ---", rs...])


	hl_best = Highlighter(f = (data, i, j) -> (i < size(ultimate, 1)) && (data[i,j]  == maximum(mask_nan_max, ultimate[i, 2:end])),
	                        crayon = crayon"yellow bold")
	hl_best_rank = Highlighter(
			f = (data, i, j) -> i == size(ultimate, 1) && (data[i,j] == minimum(ultimate[i, 2:end])),
			crayon = crayon"green bold")

	pretty_table(
		ultimate, 
		formatters = ft_printf("%.2f"),
		highlighters = (hl_best, hl_best_rank),
		body_hlines = [size(ultimate, 1) - 1]
	)
end

### test code
DrWatson.projectdir() = "/home/skvarvit/generativead/GenerativeAD.jl"
source_prefix, target_prefix = "experiments/tabular", "evaluation-pat/tabular"
generate_stats(source_prefix, target_prefix, force=true)

df = collect_stats(target_prefix)
df_agg = aggregate_stats(df, :pat_10, [:pat_10, :parameters])
df_agg[:, :pat_10] = round.(df_agg[:, :pat_10], digits=2)
print_table(df_agg, :pat_10)