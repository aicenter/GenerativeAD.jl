using DrWatson
using FileIO, BSON, DataFrames
using PrettyTables
using Statistics
using StatsBase
import PGFPlots
include("./utils/pgf_boxplot.jl")
include("./utils/ranks.jl")

using GenerativeAD.Evaluation: MODEL_MERGE, MODEL_ALIAS, DATASET_ALIAS, MODEL_TYPE, apply_aliases!
using GenerativeAD.Evaluation: _prefix_symbol, PAT_METRICS, aggregate_stats_mean_max, aggregate_stats_max_mean
using GenerativeAD.Evaluation: rank_table, print_rank_table, latex_booktabs, convert_anomaly_class

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

            filename = "$(projectdir())/paper/tables/tabular_$(metric)_$(metric)_$(name)$(suffix).tex"
            open(filename, "w") do io
                print_rank_table(io, rt; backend=:tex)
            end
        end
    end
end

basic_tables_tabular(copy(df_tabular))
basic_tables_tabular(copy(df_tabular_ens), suffix="_ensembles")


function plot_knowledge_tabular(df; suffix="", format="pdf")
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
                    ylabel="avg. rnk", ymax=6.0,
                    style="xtick={1, 2, 3, 4, 5, 6}, 
                        xticklabels={\$PR@1\$, \$PR@5\$, \$PR@10\$, \$PR@20\$, \$$(mn)_{val}\$, \$$(mn)_{tst}\$},
                        x tick label style={rotate=50,anchor=east}"
                        )
            filename = "$(projectdir())/paper/figures/tabular_knowledge_rank_$(metric)_$(name)$(suffix).$(format)"
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
            filename = "$(projectdir())/paper/figures/tabular_knowledge_rank_$(metric)_$(name)$(suffix).$(format)"
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
        
    filename = "$(projectdir())/paper/tables/tabular_aggcomp_$(metric)$(suffix).tex"
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
        
    filename = "$(projectdir())/paper/tables/tabular_metriccomp_$(name)$(suffix).tex"
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

    filename = "$(projectdir())/paper/tables/tabular_ensemblecomp_$(metric)$(suffix).tex"
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
    filename = "$(projectdir())/paper/tables/tabular_ensemblecomp_$(metric)_detail$(suffix).html"
    open(filename, "w") do io
        print_rank_table(io, rt; backend=:html)
    end
end


comparison_tabular_ensemble(copy(df_tabular), copy(df_tabular_ens), ("AUC", :auc))
comparison_tabular_ensemble(copy(df_tabular), copy(df_tabular_ens), ("TPR@5", :tpr_5))

# join baseline and ensembles, allows to dig into where they help
# compare only those models for which ensembles are computed
baseline = copy(df_tabular);
models = unique(df_tabular_ens.modelname);
ensembles = copy(vcat(filter(x -> (x.modelname in models), baseline), df_tabular_ens));
comparison_tabular_ensemble(
    baseline, 
    ensembles, ("AUC", :auc); 
    suffix="_only_improve")
comparison_tabular_ensemble(
    baseline, 
    ensembles, ("TPR@5", :tpr_5);
    suffix="_only_improve")


# training time rank vs avg rank
function plot_tabular_fit_time(df; time_col=:fit_t, suffix="", format="pdf")
    apply_aliases!(df, col="modelname", d=MODEL_MERGE)
    df["model_type"] = copy(df["modelname"])
    apply_aliases!(df, col="model_type", d=MODEL_TYPE)
    apply_aliases!(df, col="modelname", d=MODEL_ALIAS)
    # training time with two stage methods shows only the second stage
    # once the encoding fit time is added to the dataframe add it to the fit column
    # filter!(x -> (x.model_type != "two-stage"), df)

    # add first stage time for encoding and training of encoding
    df["fit_t"] .+= df["fs_fit_t"] .+ df["fs_fit_t"]
    # add all eval times together
    df["total_eval_t"] = df["tr_eval_t"] .+ df["tst_eval_t"] .+ df["val_eval_t"]

    for (mn, metric) in zip(["AUC", "TPR@5"],[:auc, :tpr_5])
        val_metric = _prefix_symbol("val", metric)
        tst_metric = _prefix_symbol("tst", metric)

        for (name, agg) in zip(
                    ["maxmean", "meanmax"], 
                    [aggregate_stats_max_mean, aggregate_stats_mean_max])

            df_agg = agg(df, val_metric, add_col=:total_eval_t)
            rt = rank_table(df_agg, tst_metric) 
            
            # time is computed only for the best models
            # maybe should be averaged over all samples
            df_agg[time_col] .= -df_agg[time_col]
            rtt = rank_table(df_agg, time_col)

            models = names(rt)[2:end]
            x = Vector(rt[end, 2:end])
            y = Vector(rtt[end, 2:end])
            # labels cannot be shifted uniformly
            a = PGFPlots.Axis([
                    PGFPlots.Plots.Scatter(x, y),
                    [PGFPlots.Plots.Node(m, xx + 0.5, yy + 0.7) for (m, xx, yy) in zip(models, x, y)]...],
                    xlabel="avg. rnk",
                    ylabel="avg. time rnk")
            filename = "$(projectdir())/paper/figures/tabular_$(time_col)_vs_$(metric)_$(name)$(suffix).$(format)"
            PGFPlots.save(filename, a; include_preamble=false)
        end
    end
end

plot_tabular_fit_time(copy(df_tabular); time_col=:fit_t, format="tex")
plot_tabular_fit_time(copy(df_tabular); time_col=:tr_eval_t, format="tex")
plot_tabular_fit_time(copy(df_tabular); time_col=:val_eval_t, format="tex")
plot_tabular_fit_time(copy(df_tabular); time_col=:tst_eval_t, format="tex")
plot_tabular_fit_time(copy(df_tabular); time_col=:total_eval_t, format="tex")
# plot_tabular_fit_time(copy(filter(x -> (x.modelname in Set(["ocsvm", "wae", "MAF", "fmgan"])), df_tabular)); time_col=:fit_t, format="pdf")

# show basic table only for autoencoders
function basic_tables_tabular_autoencoders(df; suffix="")
    df["model_type"] = copy(df["modelname"])
    apply_aliases!(df, col="model_type", d=MODEL_TYPE)
    filter!(x -> (x.model_type == "autoencoders"), df)
    apply_aliases!(df, col="modelname", d=MODEL_ALIAS)
    apply_aliases!(df, col="dataset", d=DATASET_ALIAS)

    # define further splitting of models based on parameters
    jc_mask = occursin.("jacodeco", df.parameters)
    df[jc_mask, :modelname] .=  df[jc_mask, :modelname] .*"-jc"

    lm_mask = occursin.("latent-mean", df.parameters)
    df[lm_mask, :modelname] .=  "disregard"

    l_mask = occursin.("latent_", df.parameters)
    df[l_mask, :modelname] .=  "disregard"

    ls_mask = occursin.("latent-sampled", df.parameters)
    df[ls_mask, :modelname] .=  "disregard"

    rm_mask = occursin.("reconstruction-mean", df.parameters)
    df[rm_mask, :modelname] .=  df[rm_mask, :modelname] .*"-rm"

    r_mask = occursin.("reconstruction_", df.parameters)
    df[r_mask, :modelname] .=  df[r_mask, :modelname] .*"-r"

    rs_mask = occursin.("reconstruction-sampled", df.parameters)
    df[rs_mask, :modelname] .=  df[rs_mask, :modelname] .*"-rs"

    d_mask = occursin.("disc_", df.parameters)
    df[d_mask, :modelname] .=  df[d_mask, :modelname] .*"-d"

    filter!(x -> (x.modelname != "disregard"), df)

    for metric in [:auc, :tpr_5]
        for (name, agg) in zip(
                    ["maxmean", "meanmax"], 
                    [aggregate_stats_max_mean, aggregate_stats_mean_max])
            val_metric = _prefix_symbol("val", metric)
            tst_metric = _prefix_symbol("tst", metric)    

            df_agg = agg(df, val_metric)
            sort!(df_agg, (:dataset, :modelname))
            rt = rank_table(df_agg, tst_metric)

            rt[end-2, 1] = "\$\\sigma_1\$"
            rt[end-1, 1] = "\$\\sigma_{10}\$"
            rt[end, 1] = "rnk"
            
            filename = "$(projectdir())/paper/tables/tabular_ae_only_$(metric)_$(metric)_$(name)$(suffix).tex"
            open(filename, "w") do io
                print_rank_table(io, rt; backend=:tex)
            end

            # due to its width we opted for box plot
            # recompute ranks for each dataset
            ranks = compute_ranks(rt[1:end-3, 2:end])
            models = names(rt)[2:end]

            # reverse eng. of groups
            groups = ones(Int, length(models))
            groups[startswith.(models, "aae")] .= 1
            groups[startswith.(models, "avae")] .= 2
            groups[startswith.(models, "gano")] .= 3
            groups[startswith.(models, "vae-")] .= 4
            groups[startswith.(models, "vaef")] .= 5
            groups[startswith.(models, "vaes")] .= 6
            groups[startswith.(models, "wae")] .= 7            

            # compute statistics for boxplot
            rmin, rmax = maximum(ranks, dims=2), maximum(ranks, dims=2)
            rmean = mean(ranks, dims=2)
            rmedian = median(ranks, dims=2)
            rlowq = quantile.(eachrow(ranks), 0.25)
            rhighq = quantile.(eachrow(ranks), 0.75)

            a = pgf_boxplot_grouped(rlowq, rhighq, rmedian, rmin, rmax, models, groups; h="10cm", w="8cm")
            filename = "$(projectdir())/paper/figures/tabular_ae_only_box_$(metric)_$(name)$(suffix).tex"
            open(filename, "w") do f
                write(f, a)
            end
        end
    end
end

basic_tables_tabular_autoencoders(copy(df_tabular))

######################################################################################
#######################               IMAGES                ##########################
######################################################################################
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

            filename = "$(projectdir())/paper/tables/images_$(metric)_$(metric)_$(name)$(suffix).tex"
            open(filename, "w") do io
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

    filename = "$(projectdir())/paper/tables/images_ensemblecomp_$(metric)$(suffix).tex"
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
    filename = "$(projectdir())/paper/tables/images_ensemblecomp_$(metric)_detail$(suffix).html"
    open(filename, "w") do io
        print_rank_table(io, rt; backend=:html)
    end
    ###
end

comparison_images_ensemble(copy(df_images), copy(df_images_ens), ("AUC", :auc))
comparison_images_ensemble(copy(df_images), copy(df_images_ens), ("TPR@5", :tpr_5))

# join baseline and ensembles, allows to dig into where they help
# compare only those models for which ensembles are computed
baseline_img = copy(df_images);
models_img = unique(df_images_ens.modelname);
ensembles_img = copy(vcat(filter(x -> (x.modelname in models_img), baseline_img), df_images_ens));
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
    for d in unique(dff.dataset)
        mask = (dff.dataset .== d)
        dff[mask, :dataset] .= dff[mask, :dataset] .* ":" .* convert_anomaly_class.(dff[mask, :anomaly_class], d)
    end
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

        filename = "$(projectdir())/paper/tables/images_per_ac_$(metric)_$(metric)$(suffix).tex"
        open(filename, "w") do io
            print_rank_table(io, rt; backend=:tex)
        end
    end
end

basic_tables_images_per_ac(copy(df_images))
basic_tables_images_per_ac(copy(df_images_ens), suffix="_ensembles")