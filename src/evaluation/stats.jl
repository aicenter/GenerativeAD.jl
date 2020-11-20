using DrWatson
using BSON
using FileIO
using Random
using DataFrames
using Statistics
using StatsBase
using EvalMetrics

# metric names and settings 
const BASE_METRICS = ["auc", "auprc", "tpr_5", "f1_5"]
const PAT_METRICS = ["pat_001", "pat_01", "pat_1", "pat_5", "pat_10", "pat_20"]
const PATN_METRICS = ["patn_5", "patn_10", "patn_50", "patn_100", "patn_500", "patn_1000"]
const PAC_METRICS = ["pac_5", "pac_10", "pac_50", "pac_100", "pac_500", "pac_1000"]
const TRAIN_EVAL_TIMES = ["fit_t", "tr_eval_t", "tst_eval_t", "val_eval_t"]

"""
	_prefix_symbol(prefix, s)

Modifies symbol `s` by adding `prefix` with underscore.
"""
_prefix_symbol(prefix, s) = Symbol("$(prefix)_$(s)")


"""
	_precision_at(p, labels, scores)

Computes precision on portion `p` samples with highest score.
Assumes such portion of highest scoring samples is labeled positive by the model.
"""
function _precision_at(p, labels, scores)
	pN = floor(Int, p*length(labels))
	if pN > 0
		sp = sortperm(scores, rev=true)[1:pN]
		# @info sp scores[sp] labels[sp]
		return EvalMetrics.precision(labels[sp], ones(eltype(labels), pN))
	else
		return NaN
	end
end

"""
	_nprecision_at(n, labels, scores; p=0.2)

Computes precision on `n` samples but up to `p` portion of the samples with highest score.
Assumes such highest scoring samples are labeled positive by the model.
"""
function _nprecision_at(n, labels, scores; p=0.2)
	N = length(labels)
	pN = floor(Int, p*N)
	sp = sortperm(scores, rev=true)
	if n < pN
		return EvalMetrics.precision(labels[sp[1:n]], ones(eltype(labels), n))
	else
		return EvalMetrics.precision(labels[sp[1:pN]], ones(eltype(labels), pN))
	end
end

"""
	_auc_at(n, labels, scores, auc)

Computes area under roc curve on `n` samples with highest score.
If `n` is greater than sample size the provided `auc` value is returned.
"""
function _auc_at(n, labels, scores, auc)
	if n < length(labels)
		sp = sortperm(scores, rev=true)[1:n]
		l, s = labels[sp], scores[sp]
		if all(l .== 1.0)
			return 1.0 # zooming at highest scoring samples left us with positives
		elseif all(l .== 0.0)
			return 0.0 # zooming at highest scoring samples left us with negatives
        else
            try
                roc = EvalMetrics.roccurve(l, s)
                return EvalMetrics.auc_trapezoidal(roc...)
            catch
                return NaN
            end
		end
	end
	auc
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
		fit_t = r[:fit_t],
		tr_eval_t = r[:tr_eval_t],
		tst_eval_t = r[:tst_eval_t],
		val_eval_t = r[:val_eval_t],
		seed = r[:seed])
	
	if Symbol("anomaly_class") in keys(r)
		row = merge(row, (anomaly_class = r[:anomaly_class],))
	elseif Symbol("ac") in keys(r)
		row = merge(row, (anomaly_class = r[:ac],))
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
				@warn "Invalid stats for $(f) \t $(ninvalid) | $(invrat) | $(length(scores)) | $(get(cml, 1.0, 0)) | $(get(cml, 0.0, 0))"

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

			# compute precision on most anomalous samples
			pat = [_precision_at(p/100.0, labels, scores) for p in [0.01, 0.1, 1.0, 5.0, 10.0, 20.0]]
			row = merge(row, (;zip(_prefix_symbol.(splt, PAT_METRICS), pat)...))

			patn = [_nprecision_at(n, labels, scores) for n in [5, 10, 50, 100, 500, 1000]]
			row = merge(row, (;zip(_prefix_symbol.(splt, PATN_METRICS), patn)...))	

			pac = [_auc_at(n, labels, scores, auc) for n in [5, 10, 50, 100, 500, 1000]]
			row = merge(row, (;zip(_prefix_symbol.(splt, PAC_METRICS), pac)...))	
		else
			error("$(splt)_scores contain only one value")
		end
	end

	DataFrame([row])
end

"""
	aggregate_stats_mean_max(df::DataFrame, criterion_col=:val_auc; 
								min_samples=("anomaly_class" in names(df)) ? 10 : 3, 
								downsample=Dict(), add_col=nothing)

Agregates eval metrics by seed/anomaly class over a given hyperparameter and then chooses best
model based on `criterion_col`. The output is a DataFrame of maximum #datasets*#models rows with
columns of different types
- identifiers - `dataset`, `modelname`, `phash`, `parameters`
- averaged metrics - both from test and validation data such as `tst_auc`, `val_pat_10`, etc.
- std of best hyperparameter computed for each metric over different seeds, suffixed `_std`
- std of best 10 hyperparameters computed over averaged metrics, suffixed `_top_10_std`
- samples involved in the aggregation, 
	+ `psamples` - number of runs of the best hyperparameter
	+ `dsamples` - number of sampled hyperparameters
	+ `dsamples_valid` - number of sampled hyperparameters with enough runs
When nonempty `downsample` dictionary is specified, the entries of`("model" => #samples)`, specify
how many samples should be taken into acount. These are selected randomly with fixed seed.
Optional arg `min_samples` specifies how many seed/anomaly_class combinations should be present
in order for the hyperparameter's results be considered statistically significant.
Optionally with argument `add_col` one can specify additional column to average values over.
"""
function aggregate_stats_mean_max(df::DataFrame, criterion_col=:val_auc; 
							min_samples=("anomaly_class" in names(df)) ? 10 : 3, 
							downsample=Dict(), add_col=nothing)
	agg_cols = vcat(_prefix_symbol.("val", BASE_METRICS), _prefix_symbol.("tst", BASE_METRICS))
	agg_cols = vcat(agg_cols, _prefix_symbol.("val", PAT_METRICS), _prefix_symbol.("tst", PAT_METRICS))
	agg_cols = vcat(agg_cols, _prefix_symbol.("val", PATN_METRICS), _prefix_symbol.("tst", PATN_METRICS))
	agg_cols = vcat(agg_cols, _prefix_symbol.("val", PAC_METRICS), _prefix_symbol.("tst", PAC_METRICS))
	agg_cols = vcat(agg_cols, Symbol.(TRAIN_EVAL_TIMES))
	agg_cols = (add_col !== nothing) ? vcat(agg_cols, add_col) : agg_cols
	top10_std_cols = _prefix_symbol.(agg_cols, "top_10_std")

	# agregate by seed over given hyperparameter and then choose best
	results = []
	for (dkey, dg) in pairs(groupby(df, :dataset))
		for (mkey, mg) in pairs(groupby(dg, :modelname))
			n = length(unique(mg.phash))
			# downsample models given by the `downsample` dictionary
			Random.seed!(42)
			pg = (mkey.modelname in keys(downsample)) && (downsample[mkey.modelname] < n) ? 
					groupby(mg, :phash)[randperm(n)[1:downsample[mkey.modelname]]] : 
					groupby(mg, :phash)
			Random.seed!()
			
			# filter only those hyperparameter that have sufficient number of samples
			mg_suff = reduce(vcat, [g for g in pg if nrow(g) >= min_samples])
			
			# for some methods and threshold the data frame is empty
			if nrow(mg_suff) > 0
				# aggregate over the seeds
				pg_agg = combine(groupby(mg_suff, :phash), 
							nrow => :psamples, 
							agg_cols .=> mean .=> agg_cols, 
							agg_cols .=> std, 
							:parameters => unique => :parameters) 
				
				# sort by criterion_col
				sort!(pg_agg, order(criterion_col, rev=true))
				best = first(pg_agg, 1)

				# add std of top 10 models metrics
				best_10_std = combine(first(pg_agg, 10), agg_cols .=> std .=> top10_std_cols)
				best = hcat(best, best_10_std)
				
				# add grouping keys
				best[:dataset] = dkey.dataset
				best[:modelname] = mkey.modelname
				best[:dsamples] = n
				best[:dsamples_valid] = nrow(pg_agg)

				push!(results, best)
			end
		end
	end
	vcat(results...)
end


"""
aggregate_stats_max_mean(df::DataFrame, criterion_col=:val_auc; 
							downsample=Dict(), add_col=nothing)

Chooses the best hyperparameters for each seed/anomaly_class combination by `criterion_col`
and then aggregates the metrics over seed/anomaly_class to get the final results. The output 
is a DataFrame of maximum #datasets*#models rows with
columns of different types
- identifiers - `dataset`, `modelname`
- averaged metrics - both from test and validation data such as `tst_auc`, `val_pat_10`, etc.
- std of best hyperparameters computed for each metric over different seeds, suffixed `_std`
- std of best 10 hyperparameters in each seed then averaged over seeds, suffixed `_top_10_std`
When nonempty `downsample` dictionary is specified, the entries of`("model" => #samples)`, specify
how many samples should be taken into acount. These are selected randomly with fixed seed.
As oposed to mean-max aggregation the output does not contain parameters and phash.
Optionally with argument `add_col` one can specify additional column to average values over.
"""
function aggregate_stats_max_mean(df::DataFrame, criterion_col=:val_auc; 
									downsample=Dict(), add_col=nothing)
	agg_cols = vcat(_prefix_symbol.("val", BASE_METRICS), _prefix_symbol.("tst", BASE_METRICS))
	agg_cols = vcat(agg_cols, _prefix_symbol.("val", PAT_METRICS), _prefix_symbol.("tst", PAT_METRICS))
	agg_cols = vcat(agg_cols, _prefix_symbol.("val", PATN_METRICS), _prefix_symbol.("tst", PATN_METRICS))
	agg_cols = vcat(agg_cols, _prefix_symbol.("val", PAC_METRICS), _prefix_symbol.("tst", PAC_METRICS))
	agg_cols = vcat(agg_cols, Symbol.(TRAIN_EVAL_TIMES))
	agg_cols = (add_col !== nothing) ? vcat(agg_cols, add_col) : agg_cols
	top10_std_cols = _prefix_symbol.(agg_cols, "top_10_std")

	agg_keys = ("anomaly_class" in names(df)) ? [:seed, :anomaly_class] : [:seed]
	# choose best for each seed/anomaly_class cobination and average over them
	results = []
	for (dkey, dg) in pairs(groupby(df, :dataset))
		for (mkey, mg) in pairs(groupby(dg, :modelname))
			partial_results = []

			# iterate over seed-anomaly_class groups
			for (skey, sg) in pairs(groupby(mg, agg_keys))
				n = nrow(sg)
				# downsample the number of hyperparameter if needed
				Random.seed!(42)
				ssg = (mkey.modelname in keys(downsample)) && (downsample[mkey.modelname] < n) ? 
						sg[randperm(n)[1:downsample[mkey.modelname]], :] : sg
				Random.seed!()
				
				sssg = sort(ssg, order(criterion_col, rev=true))
				# best hyperparameter after sorting by criterion_col
				best = first(sssg, 1)
				
				# add std of top 10 models metrics
				best_10_std = combine(first(sssg, 10), agg_cols .=> std .=> top10_std_cols)
				best = hcat(best, best_10_std)
				
				push!(partial_results, best)
			end

			best_per_seed = reduce(vcat, partial_results)
			# average over seed-anomaly_class groups
			best = combine(best_per_seed,  
						agg_cols .=> mean .=> agg_cols, 
						top10_std_cols .=> mean .=> top10_std_cols,
						agg_cols .=> std) 
			
			# add grouping keys
			best[:dataset] = dkey.dataset
			best[:modelname] = mkey.modelname
		
			push!(results, best)
		end
	end
	vcat(results...)
end
