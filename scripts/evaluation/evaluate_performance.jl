using ArgParse
using DrWatson
@quickactivate
using CSV
using BSON
using Random
using FileIO
using DataFrames
using PrettyTables
using PrettyTables.Crayons

using GenerativeAD.Evaluation: _prefix_symbol, PAT_METRICS, aggregate_stats_mean_max
using GenerativeAD.Evaluation: rank_table, print_rank_table

s = ArgParseSettings()
@add_arg_table! s begin
	"filename"
		arg_type = String
		default = "evaluation/images_eval.bson"
		help = "Location of cached DataFrame."
	"-c", "--criterion-metric"
		arg_type = String
		default = "val_auc"
		help = "Criterion to sort the models."
	"-r", "--rank-metric"
		arg_type = String
		default = ""
		help = "Metric to rank models."
	"-b", "--backend"
		arg_type = String
		default = "txt"
		help = "Backend for PrettyTable print. Either of [txt (default), tex, html] is allowed."
	"-o", "--output-prefix"
		arg_type = String
		default = "evaluation/images_eval"
		help = "Output prefix for storing results."
	"-v", "--verbose"
		action = :store_true
		help = "Print all results instead of storing to files."
	"-p", "--proportional"
		action = :store_true
		help = "Overloads criterion and uses pat@x% with incresing x. Prints only ranks."
	"--best-params"
		action = :store_true
		help = "Stores CSV files for each model's best parameters."
end

function main(args)
	f = datadir(args["filename"])
	df = load(f)[:df]
	@info "Loaded $(nrow(df)) rows from $f"

	if args["proportional"]
		ranks = []
		if args["rank-metric"] != ""
			for criterion in _prefix_symbol.("val", PAT_METRICS)
				df_agg = aggregate_stats_mean_max(df, criterion)
				push!(ranks, rank_table(df_agg, args["rank-metric"])[end:end, :])
			end
		else # pat/pat scenario if no rank-metric is provided
			for (criterion, metric) in zip(_prefix_symbol.("val", PAT_METRICS), _prefix_symbol.("tst", PAT_METRICS))
				df_agg = aggregate_stats_mean_max(df, criterion)
				push!(ranks, rank_table(df_agg, metric)[end:end, :])
			end
		end

		df_ranks = vcat(ranks...)
		df_ranks[:, :dataset] .= PAT_METRICS
		hl_best_rank = Highlighter(
					f = (data, i, j) -> (data[i,j] == minimum(df_ranks[i, 2:end])),
					crayon = crayon"green bold")
		
		rank_metric = (args["rank-metric"] != "") ? args["rank-metric"] : "tst_pat_x"
		
		@info "Best models chosen by validation pat@x% with incresing x"
		@info "Ranking by $(rank_metric)"
		
		if ~args["verbose"]
			target_filename = datadir(args["output-prefix"],"ranks_pat_$(rank_metric).txt")
			(~isdir(dirname(target_filename))) && mkdir(dirname(target_filename))
			open(target_filename, "w") do io
				pretty_table(
					io,
					df_ranks,
					formatters = ft_round(2),
					highlighters = (hl_best_rank),
				)
			end
		else
			pretty_table(
					df_ranks,
					formatters = ft_round(2),
					highlighters = (hl_best_rank),
				)
		end
	else
		df_agg = aggregate_stats_mean_max(df, Symbol(args["criterion-metric"]))
		rt = rank_table(df_agg, args["rank-metric"])

		@info "Best models chosen by $(args["criterion-metric"])"
		@info "Ranking by $(args["rank-metric"])"
		if ~args["verbose"]
			target_filename = datadir(args["output-prefix"],"$(args["criterion-metric"])_$(args["rank-metric"]).$(args["backend"])")
			(~isdir(dirname(target_filename))) && mkdir(dirname(target_filename))
			open(target_filename, "w") do io
				print_rank_table(io, rt; backend=Symbol(args["backend"]))
			end
		else
			print_rank_table(rt; backend=Symbol(args["backend"]))
		end

		if args["best-params"]
			target_dir = datadir(args["output-prefix"], "params_"*args["criterion-metric"])
			@info "Storing best parameters as separate CSVs in $(target_dir)"
			(~isdir(target_dir)) && mkdir(target_dir)
			all_models = unique(df_agg[:modelname])
			for m in all_models
				magg = filter(x -> (x.modelname == m), df_agg)
				CSV.write(joinpath(target_dir, "$(m).csv"), magg)
			end
		end
	end
end

main(parse_args(ARGS, s))
