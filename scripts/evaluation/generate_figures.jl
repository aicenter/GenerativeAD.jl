using DrWatson
using FileIO, BSON, DataFrames

using GenerativeAD.Evaluation: MODEL_MERGE, MODEL_ALIAS, DATASET_ALIAS, MODEL_TYPE, apply_aliases!
using GenerativeAD.Evaluation: _prefix_symbol, PAT_METRICS, aggregate_stats_mean_max, aggregate_stats_max_mean
using GenerativeAD.Evaluation: rank_table, print_rank_table

function basic_tables_tabular(df; suffix="")
    apply_aliases!(df, col="modelname", d=MODEL_MERGE)

    for metric in [:auc, :tpr_5]
        for (name, agg) in zip(
                    ["maxmean", "meanmax"], 
                    [aggregate_stats_max_mean, aggregate_stats_mean_max])
            val_metric = _prefix_symbol("val", metric)
            tst_metric = _prefix_symbol("tst", metric)    

            df_agg = agg(df, val_metric)

            df_agg["model_type"] = copy(df_agg["modelname"])
            apply_aliases!(df_agg, col="modelname", d=MODEL_ALIAS)
            apply_aliases!(df_agg, col="dataset", d=DATASET_ALIAS)
            apply_aliases!(df_agg, col="model_type", d=MODEL_TYPE)

            sort!(df_agg, (:dataset, :model_type, :modelname))
            rt = rank_table(df_agg, tst_metric)
            rt[end-2, 1] = "\$\\sigma_1\$"
            rt[end-1, 1] = "\$\\sigma_{10}\$"
            rt[end, 1] = "rnk"

            filename = "./paper/tables/tabular_$(metric)_$(metric)_$(name)$(suffix).tex"
            open(filename, "w") do io
                print_rank_table(io, rt; backend=:tex)
            end
        end
    end
end

f = datadir("evaluation/tabular_eval.bson")
df = load(f)[:df];
basic_tables_tabular(copy(df))

f = datadir("evaluation_ensembles/tabular_eval.bson")
df = load(f)[:df];
basic_tables_tabular(copy(df), suffix="_ensembles")


function basic_tables_images(df; suffix="")
    dff = filter(x -> x.seed == 1, df)
    apply_aliases!(dff, col="modelname", d=MODEL_MERGE)

    for metric in [:auc, :tpr_5]
        for (name, agg) in zip(
                    ["maxmean", "meanmax"], 
                    [aggregate_stats_max_mean, aggregate_stats_mean_max])
            val_metric = _prefix_symbol("val", metric)
            tst_metric = _prefix_symbol("tst", metric)    

            df_agg = agg(dff, val_metric)

            df_agg["model_type"] = copy(df_agg["modelname"])
            apply_aliases!(df_agg, col="modelname", d=MODEL_ALIAS)
            apply_aliases!(df_agg, col="dataset", d=DATASET_ALIAS)
            apply_aliases!(df_agg, col="model_type", d=MODEL_TYPE)

            sort!(df_agg, (:dataset, :model_type, :modelname))
            rt = rank_table(df_agg, tst_metric)
            rt[end-2, 1] = "\$\\sigma_1\$"
            rt[end-1, 1] = "\$\\sigma_{10}\$"
            rt[end, 1] = "rnk"

            open("./paper/tables/images_$(metric)_$(metric)_$(name)$(suffix).tex", "w") do io
                print_rank_table(io, rt; backend=:tex)
            end
        end
    end
end

f = datadir("evaluation/images_eval.bson")
df = load(f)[:df];
basic_tables_images(copy(df))

f = datadir("evaluation_ensembles/images_eval.bson")
df = load(f)[:df];
basic_tables_images(copy(df), suffix="_ensembles")


import PGFPlots
function plot_knowledge_tabular(df; suffix="", format="pdf")
    # filter!(x -> (x.modelname != "vae_ocsvm") && (x.modelname != "vae+ocsvm"), df)    
    apply_aliases!(df, col="modelname", d=MODEL_MERGE)
    apply_aliases!(df, col="modelname", d=MODEL_TYPE)

    for (mn, metric) in zip(["AUC", "TPR@5"],[:auc, :tpr_5])
        val_metric = _prefix_symbol("val", metric)
        tst_metric = _prefix_symbol("tst", metric)

        for (name, agg) in zip(
                    ["maxmean", "meanmax"], 
                    [aggregate_stats_max_mean, aggregate_stats_mean_max])

            ranks = []
            criterions = vcat(_prefix_symbol.("val", PAT_METRICS), [val_metric, tst_metric])
            for criterion in criterions
                df_agg = agg(df, criterion)
                rt = rank_table(df_agg, tst_metric)
                push!(ranks, rt[end:end, :])
            end

            df_ranks = vcat(ranks...)
            df_ranks[:, :dataset] .= String.(criterions)
            rename!(df_ranks, :dataset => :criterion)

            models = names(df_ranks)[2:end]
            a = PGFPlots.Axis([PGFPlots.Plots.Linear(
                            1:length(criterions), 
                            df_ranks[:, i + 1], 
                            legendentry=m) for (i, m) in enumerate(models)], 
                    ylabel="avg. rnk", ymax=4.5,
                    style="xtick={1, 2, 3, 4, 5, 6}, 
                        xticklabels={\$PR@1\$, \$PR@5\$, \$PR@10\$, \$PR@20\$, \$$(mn)_{val}\$, \$$(mn)_{tst}\$},
                        x tick label style={rotate=50,anchor=east}"
                        )
            filename = "./paper/figures/tabular_knowledge_rank_$(metric)_$(name)$(suffix).$(format)"
            PGFPlots.save(filename, a; include_preamble=false)
        end
    end
end


f = datadir("evaluation/tabular_eval.bson")
df = load(f)[:df];
plot_knowledge_tabular(copy(df); format="svg")
plot_knowledge_tabular(copy(df); format="tex")

f = datadir("evaluation_ensembles/tabular_eval.bson")
df = load(f)[:df];
plot_knowledge_tabular(copy(df), suffix="_ensembles", format="pdf")
plot_knowledge_tabular(copy(df), suffix="_ensembles", format="tex")

