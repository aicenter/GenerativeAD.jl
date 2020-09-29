using DrWatson
@quickactivate
using EvalMetrics
using FileIO
using BSON
using DataFrames
using PrettyTables
using PrettyTables.Crayons
using Statistics
using Base.Threads: @threads

# pkgs which come from BSONs
using ValueHistories


"""
	_prefix_symbol(prefix, s)

Modifies symbol `s` by adding `prefix` with underscore.
"""
_prefix_symbol(prefix, s) = Symbol("$(prefix)_$(String(s))")


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
		parameters = savename(r[:parameters]), 
		seed = r[:seed])
	
	if Symbol("anomaly_class") in keys(r)
		row = merge(row, (anomaly_class = r[:anomaly_class]))
	end

	for splt in ["val", "tst"]
		scores = r[_prefix_symbol(splt, :scores)]
		labels = r[_prefix_symbol(splt, :labels)]

		any(isnan.(scores)) && error("$(splt)_scores contain NaN")
	
		roc = EvalMetrics.roccurve(labels, scores)
		auc = EvalMetrics.auc_trapezoidal(roc...)
		prc = EvalMetrics.prcurve(labels, scores)
		auprc = EvalMetrics.auc_trapezoidal(prc...)

		t5 = EvalMetrics.threshold_at_fpr(labels, scores, 0.05)
		cm5 = ConfusionMatrix(labels, scores, t5)
		tpr5 = EvalMetrics.true_positive_rate(cm5)
		f5 = EvalMetrics.f1_score(cm5)

		row = merge(row, (;zip(_prefix_symbol.(splt, 
				["auc", "auprc", "tpr@5", "f1@5"]), 
				[auc, auprc, tpr5, f5])...))
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

	# run multithread map here
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

	frames = []
	# run multithread reduction with vcat here
	for f in files
		df = load(f)[:df]
		push!(frames, df)
	end
	vcat(frames...)
end

"""
	aggregate_stats(df::DataFrame, criterion_col=:val_auc, output_cols=[:tst_auc])

Chooses agregates eval metrics by seed over a given hyperparameter and then chooses best
model based on `criterion_col`. output_cols 
"""
function aggregate_stats(df::DataFrame, criterion_col=:val_auc, output_cols=[:tst_auc])
	metrics = ["auc", "auprc", "tpr@5", "f1@5"]
	agg_cols = vcat(_prefix_symbol.("val", metrics), _prefix_symbol.("tst", metrics))

	# agregate by seed over given hyperparameter and then choose best
	results = []
	for (dkey, dg) in pairs(groupby(df, :dataset))
		@info "Processing $(dkey.dataset) with $(nrow(dg)) trained models."
		for (mkey, mg) in pairs(groupby(dg, :modelname))
			@info "\t$(mkey.modelname) with $(nrow(mg)) experiments."
			pg = groupby(mg, :phash)
			pg_agg = combine(pg, nrow, agg_cols .=> mean .=> agg_cols)
			best = first(sort!(pg_agg, order(criterion_col, rev=true)))
			row = merge(
				(dataset = dkey.dataset, modelname = mkey.modelname), 
				(;zip(output_cols, [best[oc] for oc in output_cols])...))
			push!(results, DataFrame([row]))
		end
	end
	vcat(results...)
end

"""
	print_table(df::DataFrame)

Prints dataframe with columns [:modelname, :dataset] and one scalar variable column.
By default highlights maximum value in each row.
"""
function print_table(df::DataFrame)
	# check number of columns 
	metric_col = setdiff(names(df), ["modelname", "dataset"])
	(length(metric_col) != 1) && error("Only one metric can be printed in the summary table.")

	# transposing the dataframe
	results = []
	for (dkey, dg) in pairs(groupby(df, :dataset))
		models = dg[:modelname]
		metric = dg[metric_col[1]]
		row = merge(
			(dataset = dkey.dataset,),
			(;zip(Symbol.(models), metric)...))
		push!(results, DataFrame([row]))
	end

	# pretty print
	ultimate = vcat(results...)
	sort!(ultimate, :dataset)

	hl_best = Highlighter(f = (data, i, j) -> (j > 1) && (data[i,j]  == maximum(ultimate[i, 2:end])),
	                        crayon = crayon"yellow bold")
	pretty_table(
		ultimate, 
		formatters = ft_printf("%.3f"),
		highlighters = hl_best
	)
end

### test code
source_prefix, target_prefix = "experiments/tabular/", "evaluation/tabular/"
generate_stats(source_prefix, target_prefix, force=true)

df = collect_stats(target_prefix)
df_agg = aggregate_stats(df)
print_table(df_agg)
