using DrWatson
using FileIO, BSON, DataFrames
using PrettyTables
using Statistics

using GenerativeAD.Evaluation: MODEL_MERGE, MODEL_ALIAS, DATASET_ALIAS, MODEL_TYPE, apply_aliases!
using GenerativeAD.Evaluation: _prefix_symbol, PAT_METRICS, aggregate_stats_mean_max, aggregate_stats_max_mean
using GenerativeAD.Evaluation: rank_table, print_rank_table, latex_booktabs

df_tabular = load(datadir("evaluation/tabular_eval.bson"))[:df];
df_tabular_ens = load(datadir("evaluation_ensembles/tabular_eval.bson"))[:df];

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

basic_tables_tabular(copy(df_tabular))
basic_tables_tabular(copy(df_tabular_ens), suffix="_ensembles")


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


# plot_knowledge_tabular(copy(df_tabular); format="svg")
plot_knowledge_tabular(copy(df_tabular); format="tex")

# plot_knowledge_tabular(copy(df_tabular_ens), suffix="_ensembles", format="pdf")
plot_knowledge_tabular(copy(df_tabular_ens), suffix="_ensembles", format="tex")


function plot_knowledge_tabular(df, models; suffix="", format="pdf")
    filter!(x -> (x.modelname in models), df)
    apply_aliases!(df, col="modelname", d=MODEL_MERGE)
    apply_aliases!(df, col="modelname", d=MODEL_ALIAS)

    results = Dict()
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
                results[(metric, name, criterion)] = rt
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
    results
end

representatives=["ocsvm", "wae", "MAF", "fmgan"]
plot_knowledge_tabular(copy(df_tabular), representatives; 
                        format="tex", suffix="_representatives")

# representatives=["RealNVP", "sptn", "MAF"]
# results = plot_knowledge_tabular(copy(df_tabular), representatives; 
#                         format="pdf", suffix="_representatives")

# reduce(hcat, [results[(:auc, "maxmean", c)]["osvm"] for c in vcat(_prefix_symbol.("val", PAT_METRICS), [:val_auc, :tst_auc])])

function rank_comparison_agg(df, tm=("AUC", :auc); suffix="")
    mn, metric = tm
    apply_aliases!(df, col="modelname", d=MODEL_MERGE)

    ranks = []
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
        models = names(rt)[2:end]
        rt["agg"] = name

        select!(rt, vcat(["agg"], models))
        push!(ranks, rt[end:end,:])
    end
    df_ranks = reduce(vcat, ranks)

    hl_best_rank = LatexHighlighter(
                    (data, i, j) -> (data[i,j] == minimum(df_ranks[i, 2:end])),
                    ["color{red}","textbf"])
        
    filename = "./paper/tables/tabular_aggcomp_$(metric)$(suffix).tex"
    open(filename, "w") do io
        pretty_table(
            io, df_ranks,
            backend=:latex,
            formatters=ft_printf("%.1f"),
            highlighters=(hl_best_rank),
            nosubheader=true,
            tf=latex_booktabs
        )
    end
end


rank_comparison_agg(copy(df_tabular), ("AUC", :auc))
rank_comparison_agg(copy(df_tabular), ("TPR@5", :tpr_5))

function rank_comparison_metric(df, ta=("maxmean", aggregate_stats_max_mean); suffix="")
    name, agg = ta
    apply_aliases!(df, col="modelname", d=MODEL_MERGE)

    ranks = []
    for (mn, metric) in zip(["AUC", "TPR@5"],[:auc, :tpr_5])
        val_metric = _prefix_symbol("val", metric)
        tst_metric = _prefix_symbol("tst", metric)    

        df_agg = agg(df, val_metric)

        df_agg["model_type"] = copy(df_agg["modelname"])
        apply_aliases!(df_agg, col="modelname", d=MODEL_ALIAS)
        apply_aliases!(df_agg, col="dataset", d=DATASET_ALIAS)
        apply_aliases!(df_agg, col="model_type", d=MODEL_TYPE)

        sort!(df_agg, (:dataset, :model_type, :modelname))
        
        rt = rank_table(df_agg, tst_metric)
        models = names(rt)[2:end]
        rt["crit"] = mn

        select!(rt, vcat(["crit"], models))
        push!(ranks, rt[end:end,:])
    end
    df_ranks = reduce(vcat, ranks)

    hl_best_rank = LatexHighlighter(
                    (data, i, j) -> (data[i,j] == minimum(df_ranks[i, 2:end])),
                    ["color{red}","textbf"])
        
    filename = "./paper/tables/tabular_metriccomp_$(name)$(suffix).tex"
    open(filename, "w") do io
        pretty_table(
            io, df_ranks,
            backend=:latex,
            formatters=ft_printf("%.1f"),
            highlighters=(hl_best_rank),
            nosubheader=true,
            tf=latex_booktabs
        )
    end
end


rank_comparison_metric(copy(df_tabular), ("maxmean", aggregate_stats_max_mean))
rank_comparison_metric(copy(df_tabular), ("meanmax", aggregate_stats_mean_max))


function comparison_tabular_ensemble(df, df_ensemble, tm=("AUC", :auc); suffix="")
    mn, metric = tm
 
    apply_aliases!(df, col="modelname", d=MODEL_MERGE)
    apply_aliases!(df_ensemble, col="modelname", d=MODEL_MERGE)

    models = unique(df_ensemble.modelname)
    filter!(x -> x.modelname in models, df)

    val_metric = _prefix_symbol("val", metric)
    tst_metric = _prefix_symbol("tst", metric)

    function _rank(d)
        df_agg = aggregate_stats_max_mean(d, val_metric)
        df_agg["model_type"] = copy(df_agg["modelname"])
        apply_aliases!(df_agg, col="modelname", d=MODEL_ALIAS)
        apply_aliases!(df_agg, col="dataset", d=DATASET_ALIAS)
        apply_aliases!(df_agg, col="model_type", d=MODEL_TYPE)
        sort!(df_agg, (:dataset, :model_type, :modelname))
        rank_table(df_agg, tst_metric)
    end
        
    rt = _rank(df)
    rt[end, 1] = "baseline"
    rt_ensemble = _rank(df_ensemble)
    rt_ensemble[end, 1] = "ensembles"

    df_ranks = vcat(rt[end:end, :], rt_ensemble[end:end, :])
    dif = Matrix(rt_ensemble[1:end-3, 2:end]) - Matrix(rt[1:end-3, 2:end])
    mean_dif = mean(dif, dims=1)
    push!(df_ranks, ["avg. change", mean_dif...])

    hl_best_rank = LatexHighlighter(
                    (data, i, j) -> (i < 3) && (data[i,j] == minimum(df_ranks[i, 2:end])),
                    ["color{red}","textbf"])

    hl_best_dif = LatexHighlighter(
                    (data, i, j) -> (i == 3) && (data[i,j] == maximum(df_ranks[i, 2:end])),
                    ["color{blue}","textbf"])
        
    f_float = (v, i, j) -> (j > 1) && (i == 3) ? ft_printf("%.2f")(v,i,j) : ft_printf("%.1f")(v,i,j)

    filename = "./paper/tables/tabular_ensemblecomp_$(metric)$(suffix).tex"
    open(filename, "w") do io
        pretty_table(
            io, df_ranks,
            backend=:latex,
            formatters=f_float,
            highlighters=(hl_best_rank, hl_best_dif),
            nosubheader=true,
            tf=latex_booktabs
        )
    end

    ### shows better the difference
    rt[1:end-3, 2:end] .= dif
    rt[end-2, 1] = "σ"
    rt[end-1, 1] = "σ_1"
    # print rank change rather than just the baseline rank
    rt[end, 1] = "rnk. chng."
    rt[end, 2:end] .=  Vector(rt_ensemble[end, 2:end]) .- Vector(rt[end, 2:end])
    filename = "./paper/tables/tabular_ensemblecomp_$(metric)_detail$(suffix).html"
    open(filename, "w") do io
        print_rank_table(io, rt; backend=:html)
    end
    ###
end


comparison_tabular_ensemble(df_tabular, df_tabular_ens, ("AUC", :auc))
comparison_tabular_ensemble(df_tabular, df_tabular_ens, ("TPR@5", :tpr_5))

# join baseline and ensembles, allows to dig into where they help
# compare only those models for which ensembles are computed
baseline = copy(df_tabular);
models = unique(df_tabular_ens.modelname);
ensembles = vcat(filter(x -> (x.modelname in models), df_tabular), df_tabular_ens);
comparison_tabular_ensemble(
    baseline, 
    ensembles, ("AUC", :auc); 
    suffix="_only_improve")
comparison_tabular_ensemble(
    baseline, 
    ensembles, ("TPR@5", :tpr_5);
    suffix="_only_improve")


# and almost the same for images
df_images = load(datadir("evaluation/images_eval.bson"))[:df];
df_images_ens = load(datadir("evaluation_ensembles/images_eval.bson"))[:df];

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

basic_tables_images(copy(df_images))
basic_tables_images(copy(df_images_ens), suffix="_ensembles")


function comparison_images_ensemble(df, df_ensemble, tm=("AUC", :auc); suffix="")
    mn, metric = tm
 
    filter!(x -> (x.seed == 1), df)
    apply_aliases!(df, col="modelname", d=MODEL_MERGE)
    filter!(x -> (x.seed == 1), df_ensemble)
    apply_aliases!(df_ensemble, col="modelname", d=MODEL_MERGE)

    models = unique(df_ensemble.modelname)
    filter!(x -> x.modelname in models, df)

    val_metric = _prefix_symbol("val", metric)
    tst_metric = _prefix_symbol("tst", metric)

    function _rank(d)
        df_agg = aggregate_stats_max_mean(d, val_metric)
        df_agg["model_type"] = copy(df_agg["modelname"])
        apply_aliases!(df_agg, col="modelname", d=MODEL_ALIAS)
        apply_aliases!(df_agg, col="dataset", d=DATASET_ALIAS)
        apply_aliases!(df_agg, col="model_type", d=MODEL_TYPE)
        sort!(df_agg, (:dataset, :model_type, :modelname))
        rank_table(df_agg, tst_metric)
    end
        
    rt = _rank(df)
    rt[end, 1] = "baseline"
    rt_ensemble = _rank(df_ensemble)
    rt_ensemble[end, 1] = "ensembles"


    df_ranks = vcat(rt[end:end, :], rt_ensemble[end:end, :])
    dif = Matrix(rt_ensemble[1:end-3, 2:end]) - Matrix(rt[1:end-3, 2:end])
    mean_dif = mean(dif, dims=1)
    push!(df_ranks, ["avg. change", mean_dif...])
    
    hl_best_rank = LatexHighlighter(
                    (data, i, j) -> (i < 3) && (data[i,j] == minimum(df_ranks[i, 2:end])),
                    ["color{red}","textbf"])

    hl_best_dif = LatexHighlighter(
                    (data, i, j) -> (i == 3) && (data[i,j] == maximum(df_ranks[i, 2:end])),
                    ["color{blue}","textbf"])
        
    f_float = (v, i, j) -> (i == 3) ? ft_printf("%.2f")(v,i,j) : ft_printf("%.1f")(v,i,j)

    filename = "./paper/tables/images_ensemblecomp_$(metric)$(suffix).tex"
    open(filename, "w") do io
        pretty_table(
            io, df_ranks,
            backend=:latex,
            formatters=f_float,
            highlighters=(hl_best_rank, hl_best_dif),
            nosubheader=true,
            tf=latex_booktabs
        )
    end

    ### shows better the difference
    rt[1:end-3, 2:end] .= dif
    rt[end-2, 1] = "σ"
    rt[end-1, 1] = "σ_1"
    # print rank change rather than just the baseline rank
    rt[end, 1] = "rnk. chng."
    rt[end, 2:end] .=  Vector(rt_ensemble[end, 2:end]) .- Vector(rt[end, 2:end])
    filename = "./paper/tables/images_ensemblecomp_$(metric)_detail$(suffix).html"
    open(filename, "w") do io
        print_rank_table(io, rt; backend=:html)
    end
    ###
end

comparison_images_ensemble(df_images, df_images_ens, ("AUC", :auc))
comparison_images_ensemble(df_images, df_images_ens, ("TPR@5", :tpr_5))

# join baseline and ensembles, allows to dig into where they help
# compare only those models for which ensembles are computed
baseline_img = copy(df_images);
models_img = unique(df_images_ens.modelname);
ensembles_img = vcat(filter(x -> (x.modelname in models), df_images), df_images_ens);
comparison_images_ensemble(
    baseline_img, 
    ensembles_img, ("AUC", :auc); 
    suffix="_only_improve")
comparison_images_ensemble(
    baseline_img, 
    ensembles_img, ("TPR@5", :tpr_5);
    suffix="_only_improve")


# basic tables when anomaly_class is treated as separate dataset
function basic_tables_images_per_ac(df; suffix="")
    dff = filter(x -> x.seed == 1, df)
    apply_aliases!(dff, col="dataset", d=DATASET_ALIAS)
    dff[:dataset] .= lowercase.(dff[:dataset]) .* ":" .* lpad.(string.(dff[:anomaly_class]), 2, '0')
    select!(dff, Not(:anomaly_class))

    apply_aliases!(dff, col="modelname", d=MODEL_MERGE)
    for metric in [:auc, :tpr_5]
        val_metric = _prefix_symbol("val", metric)
        tst_metric = _prefix_symbol("tst", metric)    

        # does not play well with the aggregation when anomaly_class is present
        df_agg = aggregate_stats_mean_max(dff; min_samples=1)

        df_agg["model_type"] = copy(df_agg["modelname"])
        apply_aliases!(df_agg, col="modelname", d=MODEL_ALIAS)
        apply_aliases!(df_agg, col="model_type", d=MODEL_TYPE)

        sort!(df_agg, (:dataset, :model_type, :modelname))
        rt = rank_table(df_agg, tst_metric)
        rt[end-2, 1] = "\$\\sigma_1\$"
        rt[end-1, 1] = "\$\\sigma_{10}\$"
        rt[end, 1] = "rnk"

        open("./paper/tables/images_per_ac_$(metric)_$(metric)$(suffix).tex", "w") do io
            print_rank_table(io, rt; backend=:tex)
        end
    end
end

basic_tables_images_per_ac(copy(df_images))
basic_tables_images_per_ac(copy(df_images_ens), suffix="_ensembles")