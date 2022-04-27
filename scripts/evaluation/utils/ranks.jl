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

const MODEL_RENAME = Dict(
    "aae_full" => "aae",
    "wae_full" => "wae", 
    "vae_full" => "vae")

"""
	sorted_rank(d, agg; val_metric=:val_auc, tst_metric=:tst_auc, downsample=Dict{String, Int}())

Computes rank table from `d` and aggregation `agg`. Applies model and dataset aliases. 
Sorts models based on model_type. Returns the rank table and also the aggregated dataframe.
"""
function sorted_rank(d, agg, val_metric=:val_auc, tst_metric=:tst_auc, downsample=Dict{String, Int}(); verbose=true)
	df_agg = agg(d, val_metric; downsample=downsample, verbose=verbose)
	
	df_agg["model_type"] = copy(df_agg["modelname"])
	apply_aliases!(df_agg, col="modelname", d=MODEL_RENAME)
	apply_aliases!(df_agg, col="modelname", d=MODEL_ALIAS)
	apply_aliases!(df_agg, col="dataset", d=DATASET_ALIAS)
	apply_aliases!(df_agg, col="model_type", d=MODEL_TYPE)
	
	sort!(df_agg, (:dataset, :model_type, :modelname))
	rt = rank_table(df_agg, tst_metric)
	names(rt)[2:end], rt
end

"""
automatic aggregation of combined dataframes for single/multi class datasets

single class datasets are marked with anomaly_class = -1 and will be aggregated usin aggregate_stats_mean_max
multi class datasets are marked with anomaly_class > 0 and will be aggregated usin aggregate_stats_max_mean


"""
function aggregate_stats_auto(df::DataFrame, criterion_col=:val_auc; kwargs...)
	df_single = select(filter(x -> x.anomaly_class == -1, df), Not(:anomaly_class))
	df_multi = filter(x -> x.anomaly_class > 0, df)

	if nrow(df_single) > 0 && nrow(df_multi) > 0
		df_agg_single = aggregate_stats_mean_max(df_single, criterion_col; kwargs...)
		df_agg_multi = aggregate_stats_max_mean(df_multi, criterion_col; kwargs...)
		return	vcat(df_agg_single, df_agg_multi, cols=:intersect)
	elseif nrow(df_single) > 0
		return aggregate_stats_mean_max(df_single, criterion_col; kwargs...)
	else
		return aggregate_stats_max_mean(df_multi, criterion_col; kwargs...)
	end
end


# top 10 mean-max
function aggregate_stats_mean_max_top_10(df::DataFrame, 
										criterion_col=:val_auc; 
										min_samples=("anomaly_class" in names(df)) ? 10 : 3,
										add_col=nothing, verbose=true)
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
			mg_suff = reduce(vcat, [g for g in pg if nrow(g) >= min_samples])	
			pg_agg = combine(groupby(mg_suff, :phash), 
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
function aggregate_stats_max_mean_top_10(df::DataFrame, 
										criterion_col=:val_auc; add_col=nothing, verbose=true)
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

			if ("anomaly_class" in names(df))
				classes = unique(mg.anomaly_class)
				dif = setdiff(collect(1:10), classes)
				if (length(classes) < 10) && verbose
					@warn "$(mkey.modelname) - $(dkey.dataset): missing runs on anomaly_class $(dif)."
				end
			else
				seeds = unique(mg.seed)
				dif = setdiff(collect(1:5), seeds)
				if (length(seeds) < 3) && verbose
					@warn "$(mkey.modelname) - $(dkey.dataset): missing runs on seed $(dif)."
				end
			end

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

# splits single and multi class image datasets into "statistic" and "semantic" anomalies
_split_image_datasets(df, dt) = (
            filter(x -> x.dataset in dt, df), 
            filter(x -> ~(x.dataset in dt), df)
        )
