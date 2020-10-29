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

"""
	aggregate_stats(df::DataFrame, criterion_col=:val_auc; undersample=Dict("ocsvm" => 100))

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
When nonempty `undersample` dictionary is specified, the entries of`("model" => #samples)`, specify
how many samples should be taken into acount. These are selected randomly with fixed seed.
"""
function aggregate_stats(df::DataFrame, criterion_col=:val_auc; undersample=Dict("ocsvm" => 100))
	agg_cols = vcat(_prefix_symbol.("val", BASE_METRICS), _prefix_symbol.("tst", BASE_METRICS))
	agg_cols = vcat(agg_cols, _prefix_symbol.("val", PAT_METRICS), _prefix_symbol.("tst", PAT_METRICS))

	# agregate by seed over given hyperparameter and then choose best
	results = []
	for (dkey, dg) in pairs(groupby(df, :dataset))
		for (mkey, mg) in pairs(groupby(dg, :modelname))
			n = length(unique(mg.phash))
			# undersample models given by the undersample dictionary
			Random.seed!(42)
			pg = (mkey.modelname in keys(undersample)) ? groupby(mg, :phash)[randperm(n)[1:undersample[mkey.modelname]]] : groupby(mg, :phash)
			Random.seed!()
			
			# filter only those hyperparameter that have sufficient number of samples
			# with images at least 3*10 otherwise 3
			threshold = ("anomaly_class" in names(df)) ? 30 : 3
			mg_suff = reduce(vcat, [g for g in pg if nrow(g) >= threshold])
			
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
				best_10_std = combine(first(pg_agg, 10), agg_cols .=> std .=> _prefix_symbol.(agg_cols, "top_10_std"))
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
