using DataFrames
using Statistics
using StatsBase
using PrettyTables
using PrettyTables.Crayons

"""
	print_table(df::DataFrame, metric_col=:tst_auc; metric_std=true)

Prints dataframe with columns [:modelname, :dataset] and one scalar variable column
given by `metric_col` argument. By default highlights maximum value in each row.
Last row of the dataframe contains average rank of each model and if `metric_std=true`
the second to last row contains average std over all dataset.
There are three backends to choose from `:text (default)`, `:latex` and `:html`.
""" # TODO: split into two functions
function print_table(df::DataFrame, metric_col=:tst_auc; metric_std=true, backend=:text)
	# check if column names are present
	(!(String(metric_col) in names(df)) || !("modelname" in names(df)) || !("dataset" in names(df))) && error("Incorrect column names.")
	(metric_std && !(String(metric_col)*"_std" in names(df))) && error("DataFrame does not contain std for the given metric.")

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

	ultimate = vcat(results...)
	sort!(ultimate, :dataset)
	
	# add average std
	if metric_std
		std_column = String(metric_col)*"_std"
		# order must be the same as in the ultimate dataset
		mean_std = [
			mean(
				filter(y -> ~isnan(y), 
					filter(x -> (x.modelname == m), df)[std_column])) for m in names(ultimate[:,2:end])]
		push!(ultimate, ["MEAN_STD", mean_std...])
	end

	# add average rank
	mask_nan_max = (x) -> (isnan(x) ? -Inf : x)
	rs = zeros(size(ultimate, 2) - 1)
	for row in eachrow(ultimate)
		rs .+= StatsBase.competerank(mask_nan_max.(Vector(row[2:end])), rev = true)
	end
	rs ./= size(ultimate, 1)
	push!(ultimate, ["RANK", rs...])

	# add horizontal lines to separate derived statistics from the rest of the table
	hlines = metric_std ? [size(ultimate, 1) - 2, size(ultimate, 1) - 1] : [size(ultimate, 1) - 1]
	
	# highlight maximum values of metric in each row (i.e. per dataset)
	f_hl_best = (data, i, j) -> (i < size(ultimate, 1)) && (data[i,j]  == maximum(mask_nan_max, ultimate[i, 2:end]))
	
	# highlight minimum rank in last row
	f_hl_best_rank = (data, i, j) -> i == size(ultimate, 1) && (data[i,j] == minimum(ultimate[i, 2:end]))	

	if backend == :html
		hl_best = HTMLHighlighter(f_hl_best, HTMLDecoration(color = "blue", font_weight = "bold"))
		hl_best_rank = HTMLHighlighter(f_hl_best_rank, HTMLDecoration(color = "red", font_weight = "bold"))

		open("table.html", "w") do io
			pretty_table(
				io,
				ultimate,
				formatters=ft_round(2),
				highlighters=(hl_best, hl_best_rank),
				nosubheader=true,
				tf=html_minimalist
			)
		end
	elseif backend == :latex
		hl_best = LatexHighlighter(f_hl_best, ["color{blue}","textbf"])
		hl_best_rank = LatexHighlighter(f_hl_best_rank,	["color{red}","textbf"])

		open("table.tex", "w") do io
			pretty_table(
				io,
				ultimate, 
				backend=:latex,
				formatters=ft_round(2),
				highlighters=(hl_best, hl_best_rank),
				hlines=hlines
			)
		end
	else
		hl_best = Highlighter(f=f_hl_best, crayon=crayon"yellow bold")
		hl_best_rank = Highlighter(f=f_hl_best_rank, crayon=crayon"green bold")

		pretty_table(
			ultimate, 
			formatters=ft_round(2),
			highlighters=(hl_best, hl_best_rank),
			body_hlines=hlines
		)
	end
	ultimate
end