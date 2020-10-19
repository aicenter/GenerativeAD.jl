using ArgParse
using DrWatson
@quickactivate
using CSV
using BSON
using Random
using FileIO
using DataFrames
using Base.Threads: @threads
using PrettyTables
using PrettyTables.Crayons

using GenerativeAD.Evaluation: _prefix_symbol, PAT_METRICS, aggregate_stats, print_table

s = ArgParseSettings()
@add_arg_table! s begin
    "filename"
		arg_type = String
		default = "evaluation/tabular_eval.bson"
		help = "Location of cached DataFrame."
	"-cm", "--criterion-metric"
		arg_type = String
		default = "val_auc"
		help = "Criterion to sort the models."
	"-rm", "--rank-metric"
		arg_type = String
		default = "tst_auc"
		help = "Criterion to sort the models."
	"-d", "--deviation"
		action = :store_true
		help = "Mean standard deviation accross dataset is added to output table."
	"-b", "--backend"
		arg_type = String
		default
		help = "Backend for PrettyTable print. Either of [text (default), latex, html] is allowed."
	"-p", "--proportional"
		action = :store_true
		help = "Overloads criterion and uses pat@x% with incresing x."
	"-bp", "--best-params"
		action = :store_true
		help = "Stores CSV files for each model's best parameters."
end

function aggregate(df, criterion, metric)
	std_col = _prefix_symbol("std", metric)
    df_agg = aggregate_stats(
    	df, 
    	criterion, 
    	[metric, std_col, :parameters]; 
    	undersample=Dict("ocsvm" => 100))

	df_agg[:, metric] = round.(df_agg[:, metric], digits=2)
	df_agg[:, std_col] = round.(df_agg[:, std_col], digits=2)
	df_agg
end

function main(args)
	f = datadir(args["filename"])
	df = load(f)[:df]
	@info "Loaded $(nrow(df)) rows from "

	if args["proportional"]
		ranks = []
		for criterion in _prefix_symbol.(PAT_METRICS, "val")
			df_agg = aggregate(df, args["criterion-metric"], args["rank-metric"])
			push!(ranks, print_table(df_agg, args["criterion-metric"])[end:end, :])
		end

		df_ranks = vcat(ranks...)
		df_ranks[:, :dataset] .= PAT_METRICS
		hl_best_rank = Highlighter(
					f = (data, i, j) -> (data[i,j] == minimum(df_ranks[i, 2:end])),
					crayon = crayon"green bold")

		pretty_table(
				df_ranks,
				formatters = ft_round(2),
				highlighters = (hl_best_rank),
			)
	else
		df_agg = aggregate(df, args["criterion-metric"], args["rank-metric"])
		print_table(df_agg, args["rank-metric"]; metric_std=args["deviation"], backend=Symbol(args["backend"]))

		if args["best-params"]
			target_dir = datadir("evaluation/"*args["criterion-metric"])
			(~isdir(target_dir)) && mkdir(target_dir)
			all_models = unique(df_agg[:modelname])
			for m in all_models
				magg = filter(x -> (x.modelname == m), df_agg)
				CSV.write(joinpath("$(m).csv")), magg)
			end
		end
	end
end

main(parse_args(ARGS, s))
