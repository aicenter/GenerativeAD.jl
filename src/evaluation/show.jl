using DataFrames
using Statistics
using StatsBase
using PrettyTables
using PrettyTables.Crayons

"""
	print_table(df::DataFrame, metric_col=:tst_auc)

Prints dataframe with columns [:modelname, :dataset] and one scalar variable column
given by `metric_col` argument. By default highlights maximum value in each row.
Last row of the dataframe contains average rank of each model.
"""
function rank_table(df::DataFrame, metric_col=:tst_auc)
	# check if column names are present
	(!(String(metric_col) in names(df)) || !("modelname" in names(df)) || !("dataset" in names(df))) && error("Incorrect column names.")
	(!(String(metric_col)*"_std" in names(df))) && error("DataFrame does not contain std for the given metric.")

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

	rt = vcat(results...)
	sort!(rt, :dataset)
	
	# add average std
	std_column = String(metric_col)*"_std"
	# order must be the same as in the rt dataset
	mean_std = [
		mean(
			filter(y -> ~isnan(y), 
				filter(x -> (x.modelname == m), df)[std_column])) for m in names(rt[:,2:end])]
	push!(rt, ["MEAN_STD", mean_std...])

	# add average rank
	mask_nan_max = (x) -> (isnan(x) ? -Inf : x)
	rs = zeros(size(rt, 2) - 1)
	for row in eachrow(rt)
		rs .+= StatsBase.competerank(mask_nan_max.(Vector(row[2:end])), rev = true)
	end
	rs ./= size(rt, 1)
	push!(rt, ["RANK", rs...])

	rt
end


"""
	print_rank_table(rt::DataFrame; backend=:text)
	print_rank_table(io::IO, rt::DataFrame; backend=:text)

Pretty prints the rank table created by `rank_table` function either into given io or to stdout.
There are three backends to choose from `:text (default)`, `:latex` and `:html`.
"""
print_rank_table(rt::DataFrame; backend=:text) = print_rank_table(stdout, rt)

function print_rank_table(io::IO, rt::DataFrame; backend=:txt)
	mask_nan_max = (x) -> (isnan(x) ? -Inf : x)
	
	# horizontal lines to separate derived statistics from the rest of the table
	hlines = [size(rt, 1) - 2, size(rt, 1) - 1]
	
	# highlight maximum values of metric in each row (i.e. per dataset)
	f_hl_best = (data, i, j) -> (i < size(rt, 1)) && (data[i,j]  == maximum(mask_nan_max, rt[i, 2:end]))
	
	# highlight minimum rank in last row
	f_hl_best_rank = (data, i, j) -> i == size(rt, 1) && (data[i,j] == minimum(rt[i, 2:end]))	

	if backend == :html
		hl_best = HTMLHighlighter(f_hl_best, HTMLDecoration(color = "blue", font_weight = "bold"))
		hl_best_rank = HTMLHighlighter(f_hl_best_rank, HTMLDecoration(color = "red", font_weight = "bold"))

		pretty_table(
			io,	rt,
			formatters=ft_round(2),
			highlighters=(hl_best, hl_best_rank),
			nosubheader=true,
			tf=html_minimalist
		)
	elseif backend == :tex
		hl_best = LatexHighlighter(f_hl_best, ["color{blue}","textbf"])
		hl_best_rank = LatexHighlighter(f_hl_best_rank,	["color{red}","textbf"])

		pretty_table(
			io,	rt, 
			backend=:latex,
			formatters=ft_round(2),
			highlighters=(hl_best, hl_best_rank),
			hlines=hlines
		)
	else
		hl_best = Highlighter(f=f_hl_best, crayon=crayon"yellow bold")
		hl_best_rank = Highlighter(f=f_hl_best_rank, crayon=crayon"green bold")

		pretty_table(
			io, rt,
			formatters=ft_round(2),
			highlighters=(hl_best, hl_best_rank),
			body_hlines=hlines
		)
	end
end