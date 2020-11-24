# ideally this should be in the GenerativeAD pkg
function compute_ranks(rt)
	mask_nan_max = (x) -> (isnan(x) ? -Inf : x)
	rs = []
	for row in eachrow(rt)
		push!(rs, StatsBase.competerank(mask_nan_max.(Vector(row)), rev = true))
	end
	# each row represents ranks of a method
	reduce(hcat, rs)
end

# top 10 mean-max
function aggregate_stats_mean_max_top_10(df::DataFrame, criterion_col=:val_auc; add_col=nothing)
	agg_cols = vcat(_prefix_symbol.("val", BASE_METRICS), _prefix_symbol.("tst", BASE_METRICS))
	agg_cols = vcat(agg_cols, _prefix_symbol.("val", PAT_METRICS), _prefix_symbol.("tst", PAT_METRICS))
	agg_cols = vcat(agg_cols, _prefix_symbol.("val", PATN_METRICS), _prefix_symbol.("tst", PATN_METRICS))
	agg_cols = vcat(agg_cols, _prefix_symbol.("val", PAC_METRICS), _prefix_symbol.("tst", PAC_METRICS))
	agg_cols = vcat(agg_cols, Symbol.(TRAIN_EVAL_TIMES))
	agg_cols = (add_col !== nothing) ? vcat(agg_cols, add_col) : agg_cols
	top10_std_cols = _prefix_symbol.(agg_cols, "top_10_std")
	std_cols = _prefix_symbol.(agg_cols, "std")

	# agregate by seed over given hyperparameter and then choose best
	results = []
	for (dkey, dg) in pairs(groupby(df, :dataset))
		for (mkey, mg) in pairs(groupby(dg, :modelname))
			pg = groupby(mg, :phash)
			pg_agg = combine(pg, 
						nrow => :psamples, 
						agg_cols .=> mean .=> agg_cols, 
						agg_cols .=> std, 
						:parameters => unique => :parameters) 
			
			# sort by criterion_col
			sort!(pg_agg, order(criterion_col, rev=true))
			
			# take best 10 models
			best_10 = first(pg_agg, 10)
			best = combine(best_10, 
							agg_cols .=> mean .=> agg_cols, 
							std_cols .=> mean .=> std_cols,
							agg_cols .=> std  .=> top10_std_cols)

			# add grouping keys
			best[:dataset] = dkey.dataset
			best[:modelname] = mkey.modelname

			push!(results, best)
		end
	end
	vcat(results...)
end

# top 10 max-mean
function aggregate_stats_max_mean_top_10(df::DataFrame, criterion_col=:val_auc; add_col=nothing)
	agg_cols = vcat(_prefix_symbol.("val", BASE_METRICS), _prefix_symbol.("tst", BASE_METRICS))
	agg_cols = vcat(agg_cols, _prefix_symbol.("val", PAT_METRICS), _prefix_symbol.("tst", PAT_METRICS))
	agg_cols = vcat(agg_cols, _prefix_symbol.("val", PATN_METRICS), _prefix_symbol.("tst", PATN_METRICS))
	agg_cols = vcat(agg_cols, _prefix_symbol.("val", PAC_METRICS), _prefix_symbol.("tst", PAC_METRICS))
	agg_cols = vcat(agg_cols, Symbol.(TRAIN_EVAL_TIMES))
	agg_cols = (add_col !== nothing) ? vcat(agg_cols, add_col) : agg_cols
	top10_std_cols = _prefix_symbol.(agg_cols, "top_10_std")
	std_cols = _prefix_symbol.(agg_cols, "std")

	agg_keys = ("anomaly_class" in names(df)) ? [:seed, :anomaly_class] : [:seed]
	
	# choose best for each seed/anomaly_class cobination and average over them
	results = []
	for (dkey, dg) in pairs(groupby(df, :dataset))
		for (mkey, mg) in pairs(groupby(dg, :modelname))
			partial_results = []

			# iterate over seed-anomaly_class groups
			for (skey, sg) in pairs(groupby(mg, agg_keys))			
				ssg = sort(sg, order(criterion_col, rev=true))
				# best hyperparameter after sorting by criterion_col
				best_10 = first(ssg, 10)
				best = combine(best_10, 
							agg_cols .=> mean .=> agg_cols, 
							agg_cols .=> std  .=> top10_std_cols)
				
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