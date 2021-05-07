using DrWatson
using FileIO, BSON, DataFrames
using PrettyTables
using Statistics
using StatsBase
import PGFPlots
include("./utils/pgf_boxplot.jl")
include("./utils/ranks.jl")

using GenerativeAD.Evaluation: MODEL_MERGE, MODEL_ALIAS, DATASET_ALIAS, MODEL_TYPE, apply_aliases!
using GenerativeAD.Evaluation: _prefix_symbol, aggregate_stats_mean_max, aggregate_stats_max_mean
using GenerativeAD.Evaluation: PAT_METRICS, PATN_METRICS, PAC_METRICS, BASE_METRICS, TRAIN_EVAL_TIMES
using GenerativeAD.Evaluation: rank_table, print_rank_table, latex_booktabs, convert_anomaly_class

const PAT_METRICS_NAMES = ["\$PR@\\%0.01\$","\$PR@\\%0.1\$","\$PR@\\%1\$","\$PR@\\%5\$","\$PR@\\%10\$","\$PR@\\%20\$"]
const PAC_METRICS_NAMES = ["\$AUC@\\#5\$","\$AUC@\\#10\$","\$AUC@\\#50\$","\$AUC@\\#100\$","\$AUC@\\#500\$","\$AUC@\\#1000\$"]
const PATN_METRICS_NAMES = ["\$PR@\\#5\$","\$PR@\\#10\$","\$PR@\\#50\$","\$PR@\\#100\$","\$PR@\\#500\$","\$PR@\\#1000\$"]

AE_MERGE = Dict(
    "aae_full" => "aae",
    "aae_vamp" => "aae",
    "wae_full" => "wae", 
    "wae_vamp" => "wae",
    "vae_full" => "vae")

DOWNSAMPLE = Dict(
    zip(["aae", "vae", "wae", "MAF", "RealNVP"], [100, 100, 100, 100, 100]))

function _merge_ae_filter!(df)
    # filter out simple vae as its properties are shown in the separate table focused on ae
    filter!(x -> (x.modelname != "vae_simple"), df)

    # filter out ocsvm_nu
    filter!(x -> (x.modelname != "ocsvm_nu"), df)

    # merge autoencoders
    apply_aliases!(df, col="modelname", d=AE_MERGE)
    
    # filter only reconstruction_samples score
    filter!(x -> ~(x.modelname in (["aae", "vae", "wae"])) || occursin("_score=reconstruction-sampled", x.parameters), df)        
    
    # filter only default parameters of "ocsvm" - rbf + nu = 0.5, tune gamma
    filter!(x -> (x.modelname != "ocsvm_rbf") || occursin("nu=0.5", x.parameters), df)
    df
end

function _filter_ensembles!(df)
    filter!(x -> occursin("ignore_nan=true_method=mean", x.parameters), df)
end

df_tabular = load(datadir("evaluation/tabular_eval.bson"))[:df];
df_tabular_clean = load(datadir("evaluation/tabular_clean_val_final_eval.bson"))[:df];
df_tabular_ens = load(datadir("evaluation_ensembles/tabular_eval.bson"))[:df];
@info "Loaded results from tabular evaluation."

function basic_summary_table(df; suffix="", prefix="", downsample=Dict{String, Int}())
    apply_aliases!(df, col="modelname", d=MODEL_MERGE)

    for metric in [:auc, :tpr_5]
        agg_names = ["maxmean", "meanmax"]
        agg_funct = [aggregate_stats_max_mean, aggregate_stats_mean_max]
        for (name, agg) in zip(agg_names, agg_funct)
            val_metric = _prefix_symbol("val", metric)
            tst_metric = _prefix_symbol("tst", metric)    

            df_agg = agg(df, val_metric, downsample=downsample)

            df_agg["model_type"] = copy(df_agg["modelname"])
            apply_aliases!(df_agg, col="modelname", d=MODEL_ALIAS)
            apply_aliases!(df_agg, col="dataset", d=DATASET_ALIAS)
            apply_aliases!(df_agg, col="model_type", d=MODEL_TYPE)

            sort!(df_agg, (:dataset, :model_type, :modelname))
            rt = rank_table(df_agg, tst_metric)
            rt[end-2, 1] = "\$\\sigma_1\$"
            rt[end-1, 1] = "\$\\sigma_{10}\$"
            rt[end, 1] = "rnk"

            filename = "$(projectdir())/paper/tables/$(prefix)_$(metric)_$(metric)_$(name)$(suffix).tex"
            open(filename, "w") do io
                print_rank_table(io, rt; backend=:tex)
            end
        end
    end
end

function basic_summary_table_tabular(df; suffix="", downsample=Dict{String, Int}())
    basic_summary_table(df; prefix="tabular", suffix=suffix, downsample=downsample)
end


# basic_summary_table_tabular(copy(df_tabular))
# basic_summary_table_tabular(_merge_ae_filter!(copy(df_tabular)), suffix="_merge_ae")
basic_summary_table_tabular(_merge_ae_filter!(copy(df_tabular)), suffix="_merge_ae_down", downsample=DOWNSAMPLE)

basic_summary_table_tabular(apply_aliases!(copy(df_tabular_clean), col="modelname", d=AE_MERGE), suffix="_clean_default")

# basic_summary_table_tabular(copy(df_tabular_ens), suffix="_ensembles")
basic_summary_table_tabular(_filter_ensembles!(copy(df_tabular_ens)), suffix="_ensembles_merge_ae_down")
@info "basic_summary_table_tabular"

# generic for both images and tabular data just change prefix and filter input
function plot_knowledge_ranks(df; prefix="tabular", suffix="", format="pdf", downsample=Dict{String, Int}())
    for (mn, metric) in zip(["AUC", "TPR@5"],[:auc, :tpr_5])
        val_metric = _prefix_symbol("val", metric)
        tst_metric = _prefix_symbol("tst", metric)

        for (name, agg) in zip(
                    ["maxmean", "meanmax"], 
                    [aggregate_stats_max_mean, aggregate_stats_mean_max])

            n = (prefix == "tabular") ? 4 : 1
            for (ctype, cnames, criterions) in zip(
                                ["pat", "pac", "patn"],
                                [PAT_METRICS_NAMES[n:end], PAC_METRICS_NAMES, PATN_METRICS_NAMES],
                                [_prefix_symbol.("val", PAT_METRICS[n:end]), _prefix_symbol.("val", PAC_METRICS), _prefix_symbol.("val", PATN_METRICS)])
                ranks = []
                metric_means = []
                extended_criterions = vcat(criterions, [val_metric, tst_metric])
                extended_cnames = vcat(cnames, ["\$$(mn)_{val}\$", "\$$(mn)_{tst}\$"])

                for criterion in extended_criterions
                    df_agg = agg(df, criterion; downsample=downsample)
                    sort!(df_agg, (order(:dataset), order(:modelname)))
                    rt = rank_table(df_agg, tst_metric)
                    push!(ranks, rt[end:end, :])
                    push!(metric_means, mean(Matrix(rt[1:end-3, 2:end]), dims=1))
                end

                df_ranks = vcat(ranks...)
                metric_mean = vcat(metric_means...)

                models = names(df_ranks)[2:end]
                
                a = PGFPlots.Axis([PGFPlots.Plots.Linear(
                                1:length(extended_criterions), 
                                df_ranks[:, i + 1]) for (i, m) in enumerate(models)], 
                        ylabel="avg. rnk",
                        style="xtick=$(_pgf_array(1:length(extended_criterions))), 
                            xticklabels=$(_pgf_array(extended_cnames)),
                            width=6cm, height=4cm, scale only axis=true,
                            x tick label style={rotate=50,anchor=east}")
                b = PGFPlots.Axis([PGFPlots.Plots.Linear(
                                1:length(extended_criterions), 
                                metric_mean[:,i], 
                                legendentry=m) for (i, m) in enumerate(models)], 
                        ylabel="avg. $mn",
                        legendStyle = "at={(0.5,1.02)}, anchor=south",
                        style="width=6cm, height=3cm, scale only axis=true, 
                        xtick=$(_pgf_array(1:length(criterions))), 
                        xticklabels={}, legend columns = -1")
                g = PGFPlots.GroupPlot(1, 2, groupStyle = "vertical sep = 0.5cm")
                push!(g, b); push!(g, a); 
                filename = "$(projectdir())/paper/figures/$(prefix)_knowledge_rank_$(ctype)_$(metric)_$(name)$(suffix).$(format)"
                PGFPlots.save(filename, g; include_preamble=false)
            end
        end
    end
end

function plot_knowledge_tabular_repre(df, models; suffix="", format="pdf", downsample=Dict{String, Int}())
    filter!(x -> (x.modelname in models), df)
    apply_aliases!(df, col="modelname", d=MODEL_MERGE)
    apply_aliases!(df, col="modelname", d=MODEL_ALIAS)
    plot_knowledge_ranks(df; prefix="tabular", suffix=suffix, format=format, downsample=downsample)
end

function plot_knowledge_tabular_type(df; suffix="", format="pdf")
    apply_aliases!(df, col="modelname", d=MODEL_MERGE)
    apply_aliases!(df, col="modelname", d=MODEL_TYPE)
    plot_knowledge_ranks(df; prefix="tabular", suffix=suffix, format=format)
end


representatives=["ocsvm", "wae", "MAF", "fmgan", "vae_ocsvm", "vae+ocsvm"]
# plot_knowledge_tabular_repre(copy(df_tabular), representatives; format="tex", suffix="_representatives")
# plot_knowledge_tabular_repre(copy(df_tabular_ens), representatives; suffix="_representatives_ensembles", format="tex")
plot_knowledge_tabular_repre(
    _merge_ae_filter!(copy(df_tabular)), representatives; suffix="_representatives_merge_ae_down", format="tex", downsample=DOWNSAMPLE)
@info "plot_knowledge_tabular_repre"

# these ones are really costly as there are 12 aggregations over all models
# plot_knowledge_tabular_type(copy(df_tabular); format="tex") 
# plot_knowledge_tabular_type(copy(df_tabular_ens), suffix="_ensembles", format="tex")
@info "plot_knowledge_tabular_type"

function rank_comparison_agg(df, tm=("AUC", :auc); prefix="tabular", suffix="")
    mn, metric = tm
    apply_aliases!(df, col="modelname", d=MODEL_MERGE)

    ranks = []
    agg_names = ["maxmean", "meanmax", "maxmean10", "meanmax10"]
    agg_funct = [aggregate_stats_max_mean, aggregate_stats_mean_max, aggregate_stats_max_mean_top_10, aggregate_stats_mean_max_top_10]
    for (name, agg) in zip(agg_names, agg_funct)
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
        
    filename = "$(projectdir())/paper/tables/$(prefix)_aggcomp_$(metric)$(suffix).tex"
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

# superseeded by per_seed table
# rank_comparison_agg(copy(df_tabular), ("AUC", :auc))
# rank_comparison_agg(copy(df_tabular), ("TPR@5", :tpr_5))
# @info "rank_comparison_agg"

function rank_comparison_metric(df, ta=("maxmean", aggregate_stats_max_mean); suffix="", downsample=Dict{String, Int}())
    name, agg = ta
    apply_aliases!(df, col="modelname", d=MODEL_MERGE)

    ranks = []
    for (mn, metric) in zip(["AUC", "TPR@5"],[:auc, :tpr_5])
        val_metric = _prefix_symbol("val", metric)
        tst_metric = _prefix_symbol("tst", metric)    

        df_agg = agg(df, val_metric, downsample=downsample)

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

# superseeded by per_seed table
# rank_comparison_metric(copy(df_tabular), ("maxmean", aggregate_stats_max_mean))
# rank_comparison_metric(copy(df_tabular), ("meanmax", aggregate_stats_mean_max))
# rank_comparison_metric(_merge_ae_filter!(copy(df_tabular)), ("maxmean", aggregate_stats_max_mean), suffix="_merge_ae_down", downsample=DOWNSAMPLE)
# rank_comparison_metric(_merge_ae_filter!(copy(df_tabular)), ("meanmax", aggregate_stats_mean_max), suffix="_merge_ae_down", downsample=DOWNSAMPLE)
# @info "rank_comparison_metric"

function comparison_ensemble(df, df_ensemble, tm=("AUC", :auc); downsample=Dict{String,Int}(), prefix="", suffix="")
    mn, metric = tm
 
    apply_aliases!(df, col="modelname", d=MODEL_MERGE)
    apply_aliases!(df_ensemble, col="modelname", d=MODEL_MERGE)

    models = unique(df_ensemble.modelname)
    filter!(x -> x.modelname in models, df)

    val_metric = _prefix_symbol("val", metric)
    tst_metric = _prefix_symbol("tst", metric)

    function _rank(d)
        df_agg = aggregate_stats_max_mean(d, val_metric; downsample=downsample)
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
    mean_dif = round.(mean(dif, dims=1), digits=2)
    push!(df_ranks, ["avg. change", mean_dif...])

    hl_best_rank = LatexHighlighter(
                    (data, i, j) -> (i < 3) && (data[i,j] == minimum(df_ranks[i, 2:end])),
                    ["color{red}","textbf"])

    hl_best_dif = LatexHighlighter(
                    (data, i, j) -> (i == 3) && (data[i,j] == maximum(df_ranks[i, 2:end])),
                    ["color{blue}","textbf"])
        
    f_float = (v, i, j) -> (j > 1) && (i == 3) ? ft_printf("%.2f")(v,i,j) : ft_printf("%.1f")(v,i,j)

    filename = "$(projectdir())/paper/tables/$(prefix)_ensemblecomp_$(metric)$(suffix).tex"
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
end

function comparison_tabular_ensemble(df, df_ensemble, tm; downsample=DOWNSAMPLE, suffix="")
    comparison_ensemble(
        df, df_ensemble, tm; 
        downsample=downsample, prefix="tabular", suffix=suffix)
end


# comparison_tabular_ensemble(copy(df_tabular), copy(df_tabular_ens), ("AUC", :auc))
# comparison_tabular_ensemble(copy(df_tabular), copy(df_tabular_ens), ("TPR@5", :tpr_5))
comparison_tabular_ensemble(
    _merge_ae_filter!(copy(df_tabular)), 
    _filter_ensembles!(copy(df_tabular_ens)), 
    ("AUC", :auc); downsample=DOWNSAMPLE, suffix="_merge_ae_down_mean")
comparison_tabular_ensemble(
    _merge_ae_filter!(copy(df_tabular)), 
    _filter_ensembles!(copy(df_tabular_ens)),
    ("TPR@5", :tpr_5); downsample=DOWNSAMPLE, suffix="_merge_ae_down_mean")
@info "comparison_tabular_ensemble"

# join baseline and ensembles, allows to dig into where they help
# compare only those models for which ensembles are computed
# baseline = copy(df_tabular);
# models = unique(df_tabular_ens.modelname);
# ensembles = copy(vcat(filter(x -> (x.modelname in models), baseline), df_tabular_ens));
# comparison_tabular_ensemble(
#     baseline, 
#     ensembles, ("AUC", :auc); 
#     suffix="_only_improve")
# comparison_tabular_ensemble(
#     baseline, 
#     ensembles, ("TPR@5", :tpr_5);
#     suffix="_only_improve")
# @info "comparison_tabular_ensemble_only_improve"

# how the choice of creation criterion and size of an ensembles affect results
function ensemble_sensitivity(df, df_ensemble; prefix="tabular", suffix="")
    apply_aliases!(df, col="modelname", d=MODEL_MERGE)
    apply_aliases!(df_ensemble, col="modelname", d=MODEL_MERGE)

    for metric in [:auc, :tpr_5]
        val_metric = _prefix_symbol("val", metric)
        tst_metric = _prefix_symbol("tst", metric)    

        function _rank(df_agg)
            df_agg["model_type"] = copy(df_agg["modelname"])
            apply_aliases!(df_agg, col="modelname", d=MODEL_ALIAS)
            apply_aliases!(df_agg, col="dataset", d=DATASET_ALIAS)
            apply_aliases!(df_agg, col="model_type", d=MODEL_TYPE)

            sort!(df_agg, (:dataset, :model_type, :modelname))
            
            _rt = rank_table(df_agg, tst_metric)
            models = names(_rt)[2:end]
            _rt, models
        end

        ranks = []

        # computing baseline
        df_agg = aggregate_stats_max_mean(df, val_metric)
        rt, models = _rank(df_agg)
        rt["criterion-size"] = "baseline"
        select!(rt, vcat(["criterion-size"], models))
        push!(ranks, rt[end:end, :])

        # computing different ensembles
        for (cn, c) in zip(["AUC", "TPR@5"], [:val_auc, :val_tpr_5])
            for s in [5, 10]
                dff = filter(x -> startswith(x.parameters, "criterion=$(c)") && endswith(x.parameters, "_size=$(s)"), df_ensemble)
                df_agg = aggregate_stats_max_mean(dff, val_metric)
                rt_ensemble, models = _rank(df_agg)
                rt_ensemble["criterion-size"] = "$(cn)-$(s)"
                select!(rt_ensemble, vcat(["criterion-size"], models))
                
                dif = Matrix(rt_ensemble[1:end-3, 2:end]) - Matrix(rt[1:end-3, 2:end])
                mean_dif = round.(mean(dif, dims=1), digits=2)

                push!(rt_ensemble, ["avg. change", mean_dif...])
                push!(ranks, rt_ensemble[end-1:end, :])
            end
        end

        df_ranks = reduce(vcat, ranks)
        hl_best_rank = LatexHighlighter(
                        (data, i, j) -> ((i == 1) || (i%2 == 0)) && (data[i,j] == minimum(df_ranks[i, 2:end])),
                        ["color{red}","textbf"])

        hl_best_dif = LatexHighlighter(
                        (data, i, j) -> (i != 1) && (i%2 == 1) && (data[i,j] == maximum(df_ranks[i, 2:end])),
                        ["color{blue}","textbf"])
    
        f_float = (v, i, j) -> (i != 1) && (i%2 == 1) ? ft_printf("%.2f")(v,i,j) : ft_printf("%.1f")(v,i,j)

        filename = "$(projectdir())/paper/tables/$(prefix)_ensemble_size_$(metric)$(suffix).tex"
        open(filename, "w") do io
            pretty_table(
                io, df_ranks,
                backend=:latex,
                formatters=f_float,
                highlighters=(hl_best_rank, hl_best_dif),
                nosubheader=true,
                tf=latex_booktabs)
        end
    end
end

ensemble_sensitivity(
    _merge_ae_filter!(copy(df_tabular)),
    _filter_ensembles!(copy(df_tabular_ens)); prefix="tabular")
@info "ensemble_sensitivity_images"

# training time rank vs avg rank
function plot_tabular_fit_time(df; time_col=:fit_t, suffix="", format="pdf", downsample=Dict{String,Int}())
    apply_aliases!(df, col="modelname", d=MODEL_MERGE)
    df["model_type"] = copy(df["modelname"])
    apply_aliases!(df, col="model_type", d=MODEL_TYPE)
    apply_aliases!(df, col="modelname", d=MODEL_ALIAS)
    
    # add first stage time for encoding and training of encoding
    df["fit_t"] .+= df["fs_fit_t"] .+ df["fs_fit_t"]
    # add all eval times together
    df["total_eval_t"] = df["tr_eval_t"] .+ df["tst_eval_t"] .+ df["val_eval_t"]

    for (mn, metric) in zip(["AUC", "TPR@5"],[:auc, :tpr_5])
        val_metric = _prefix_symbol("val", metric)
        tst_metric = _prefix_symbol("tst", metric)

        # only meanmax to be consistent
        for (name, agg) in zip(
                    ["meanmax"], 
                    [aggregate_stats_mean_max])

            df_agg = agg(df, val_metric, add_col=:total_eval_t, downsample=downsample)
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

# plot_tabular_fit_time(copy(df_tabular); time_col=:fit_t, format="tex")
# plot_tabular_fit_time(copy(df_tabular); time_col=:tr_eval_t, format="tex")
# plot_tabular_fit_time(copy(df_tabular); time_col=:val_eval_t, format="tex")
# plot_tabular_fit_time(copy(df_tabular); time_col=:tst_eval_t, format="tex")
# plot_tabular_fit_time(copy(df_tabular); time_col=:total_eval_t, format="tex")

plot_tabular_fit_time(_merge_ae_filter!(copy(df_tabular)); time_col=:fit_t, format="tex", suffix="_merge_ae_down", downsample=DOWNSAMPLE)
plot_tabular_fit_time(_merge_ae_filter!(copy(df_tabular)); time_col=:tr_eval_t, format="tex", suffix="_merge_ae_down", downsample=DOWNSAMPLE)
plot_tabular_fit_time(_merge_ae_filter!(copy(df_tabular)); time_col=:val_eval_t, format="tex", suffix="_merge_ae_down", downsample=DOWNSAMPLE)
plot_tabular_fit_time(_merge_ae_filter!(copy(df_tabular)); time_col=:tst_eval_t, format="tex", suffix="_merge_ae_down", downsample=DOWNSAMPLE)
plot_tabular_fit_time(_merge_ae_filter!(copy(df_tabular)); time_col=:total_eval_t, format="tex", suffix="_merge_ae_down", downsample=DOWNSAMPLE)
@info "plot_tabular_fit_time"

# show basic table only for autoencoders
function tabular_autoencoders_investigation(df; split_vamp=false, suffix="")
    df["model_type"] = copy(df["modelname"])
    apply_aliases!(df, col="model_type", d=MODEL_TYPE)
    filter!(x -> (x.model_type == "autoencoders"), df)
    apply_aliases!(df, col="modelname", d=MODEL_ALIAS)
    apply_aliases!(df, col="dataset", d=DATASET_ALIAS)

    # split aae_full and wae_full based on used prior
    if split_vamp
        aaefv_mask = (df.modelname .== "aaef") .& occursin.("prior=vamp", df.parameters)
        df[aaefv_mask, :modelname] .=  "aaefv"
        waefv_mask = (df.modelname .== "waef") .& occursin.("prior=vamp", df.parameters)
        df[waefv_mask, :modelname] .=  "waefv"
    else
        apply_aliases!(df, col="modelname", d=Dict("aaev" => "aae", "waev" => "wae"))
    end

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
    df[r_mask, :modelname] .=  "disregard"

    rs_mask = occursin.("reconstruction-sampled", df.parameters)
    df[rs_mask, :modelname] .=  df[rs_mask, :modelname] .*"-rs"

    d_mask = occursin.("disc_", df.parameters)
    df[d_mask, :modelname] .=  df[d_mask, :modelname] .*"-d"

    el_mask = occursin.("elbo_", df.parameters)
    df[el_mask, :modelname] .=  df[el_mask, :modelname] .*"-el"

    mse_mask = occursin.("score=mse_", df.parameters)
    df[mse_mask, :modelname] .=  df[mse_mask, :modelname] .*"-rm"

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

            #### grouping by method
            mgroups = ones(Int, length(models))
            mgroups[startswith.(models, "avae")] .= 1
            mgroups[startswith.(models, "gano")] .= 2
            mgroups[startswith.(models, "aae-")] .= 3
            mgroups[startswith.(models, "aaev")] .= 4
            mgroups[startswith.(models, "aaef-")] .= 5
            mgroups[startswith.(models, "aaefv")] .= 6
            mgroups[startswith.(models, "vae-")] .= 7
            mgroups[startswith.(models, "vaef")] .= 8
            mgroups[startswith.(models, "vaes")] .= 9
            mgroups[startswith.(models, "wae-")] .= 10
            mgroups[startswith.(models, "waev")] .= 11
            mgroups[startswith.(models, "waef-")] .= 12
            mgroups[startswith.(models, "waefv")] .= 13

            sm = sortperm(mgroups)
            mgroups = mgroups[sm]

            models = models[sm]
            ranks = ranks[sm, :]

            #### grouping by score, 5 types + 1 for other scores
            groups = 6*ones(Int, length(models))
            groups[endswith.(models, "-rm")] .= 1
            groups[endswith.(models, "-rs")] .= 2
            groups[endswith.(models, "-jc")] .= 3
            groups[endswith.(models, "-d")] .= 4
            groups[endswith.(models, "-el")] .= 5
            ####


            # compute statistics for boxplot
            rmin, rmax = maximum(ranks, dims=2), maximum(ranks, dims=2)
            rmean = mean(ranks, dims=2)
            rmedian = median(ranks, dims=2)
            rlowq = quantile.(eachrow(ranks), 0.25)
            rhighq = quantile.(eachrow(ranks), 0.75)

            a = pgf_boxplot_grouped_colorpos(rlowq, rhighq, rmedian, rmin, rmax, models, groups, mgroups; h="12cm", w="8cm")
            filename = "$(projectdir())/paper/figures/tabular_ae_only_box_$(metric)_$(name)$(suffix).tex"
            open(filename, "w") do f
                write(f, a)
            end
        end
    end
end

# tabular_autoencoders_investigation(copy(df_tabular))
tabular_autoencoders_investigation(copy(df_tabular); split_vamp=true, suffix="_split_vamp")
@info "tabular_autoencoders_investigation"

# does crossvalidation matters?
function per_seed_ranks_tabular(df; suffix="", downsample=Dict{String, Int}())
    apply_aliases!(df, col="modelname", d=MODEL_MERGE)

    for metric in [:auc, :tpr_5]
        val_metric = _prefix_symbol("val", metric)
        tst_metric = _prefix_symbol("tst", metric)    

        function _rank(df_agg)
            df_agg["model_type"] = copy(df_agg["modelname"])
            apply_aliases!(df_agg, col="modelname", d=MODEL_ALIAS)
            apply_aliases!(df_agg, col="dataset", d=DATASET_ALIAS)
            apply_aliases!(df_agg, col="model_type", d=MODEL_TYPE)

            sort!(df_agg, (:dataset, :model_type, :modelname))
            
            rt = rank_table(df_agg, tst_metric)
            models = names(rt)[2:end]
            rt, models
        end

        ranks = []
        for (name, agg) in zip(
                ["maxmean", "meanmax"], 
                [aggregate_stats_max_mean, aggregate_stats_mean_max])
            df_agg = agg(df, val_metric; downsample=downsample)
            rt, models = _rank(df_agg)
            rt["exp"] = name
            select!(rt, vcat(["exp"], models))
            push!(ranks, rt[end:end,:])
        end

        for seed in 1:5
            dff = filter(x -> (x.seed == seed), df)
            # silence the warnings for insufficient number of seeds
            df_agg = aggregate_stats_max_mean(dff, val_metric; verbose=false, downsample=downsample)
            rt, models = _rank(df_agg)
            rt["exp"] = "seed=$seed"
            select!(rt, vcat(["exp"], models))
            push!(ranks, rt[end:end,:])
        end

        df_ranks = reduce(vcat, ranks)
        hl_best_rank = LatexHighlighter(
                        (data, i, j) -> (data[i,j] == minimum(df_ranks[i, 2:end])),
                        ["color{red}","textbf"])
    
        filename = "$(projectdir())/paper/tables/tabular_per_seed_ranks_$(metric)$(suffix).tex"
        open(filename, "w") do io
            pretty_table(
                io, df_ranks,
                backend=:latex,
                formatters=ft_printf("%.1f"),
                highlighters=(hl_best_rank),
                nosubheader=true,
                tf=latex_booktabs)
        end
    end
end

# per_seed_ranks_tabular(copy(df_tabular))
# per_seed_ranks_tabular(_merge_ae_filter!(copy(df_tabular)), suffix="_merge_ae")
per_seed_ranks_tabular(_merge_ae_filter!(copy(df_tabular)), suffix="_merge_ae_down", downsample=DOWNSAMPLE)
# per_seed_ranks_tabular(copy(df_tabular_ens); suffix="_ensembles")
@info "per_seed_ranks_tabular"


######################################################################################
#######################               IMAGES                ##########################
######################################################################################
df_images = load(datadir("evaluation/images_eval.bson"))[:df];
df_images_loi = load(datadir("evaluation/images_leave-one-in_eval.bson"))[:df];
df_images_ens = load(datadir("evaluation_ensembles/images_eval.bson"))[:df];
df_images_class = load(datadir("evaluation/images_leave-one-out_eval.bson"))[:df];
@info "Loaded results from images"

function filter_img_models!(df)
    filter!(x -> (x.modelname != "aae_ocsvm") && (x.modelname != "fAnoGAN-GP"), df)
end    

function basic_summary_table_images(df; suffix="")
    filter!(x -> x.seed == 1, df)
    basic_summary_table(df; prefix="images", suffix=suffix)
end

function basic_summary_table_images_class(df; suffix="")
    select!(df, Not(:anomaly_class)) # neither mnist-c nor mvtech have only one anomaly_class
    basic_summary_table(df; prefix="images_class", suffix=suffix)
end

# basic_summary_table_images(copy(df_images))
# basic_summary_table_images(copy(df_images_ens), suffix="_ensembles")

basic_summary_table_images(filter_img_models!(copy(df_images)), suffix="_filter")
basic_summary_table_images(copy(df_images_loi), suffix="_loi")
basic_summary_table_images(filter_img_models!(copy(df_images_ens)), suffix="_ensembles_filter")
basic_summary_table_images_class(copy(df_images_class); suffix="")
@info "basic_summary_table_images"

function rank_comparison_agg_images(df, tm=("AUC", :auc); suffix="")
    filter!(x -> (x.seed == 1), df)
    rank_comparison_agg(df, tm; prefix="images", suffix=suffix)
end

# rank_comparison_agg_images(copy(df_images), ("AUC", :auc))
# rank_comparison_agg_images(copy(df_images), ("TPR@5", :tpr_5))
@info "rank_comparison_agg_images"

function comparison_images_ensemble(df, df_ensemble, tm=("AUC", :auc); suffix="")
    filter!(x -> (x.seed == 1), df)
    filter!(x -> (x.seed == 1), df_ensemble)
    comparison_ensemble(df, df_ensemble, tm; prefix="images", suffix=suffix)
end

# comparison_images_ensemble(copy(df_images), copy(df_images_ens), ("AUC", :auc))
# comparison_images_ensemble(copy(df_images), copy(df_images_ens), ("TPR@5", :tpr_5))
comparison_images_ensemble(
    filter_img_models!(copy(df_images)), 
    _filter_ensembles!(filter_img_models!(copy(df_images_ens))), 
    ("AUC", :auc); suffix="_filter")
comparison_images_ensemble(
    filter_img_models!(copy(df_images)), 
    _filter_ensembles!(filter_img_models!(copy(df_images_ens))), 
    ("TPR@5", :tpr_5); suffix="_filter")

@info "comparison_images_ensemble"

function ensemble_sensitivity_images(df, df_ensemble)
    filter!(x -> (x.seed == 1), df)
    filter!(x -> (x.seed == 1), df_ensemble)
    ensemble_sensitivity(df, df_ensemble; prefix="images")
end

ensemble_sensitivity_images(
        filter_img_models!(copy(df_images)),
        _filter_ensembles!(filter_img_models!(copy(df_images_ens))))

@info "ensemble_sensitivity_images"

# join baseline and ensembles, allows to dig into where they help
# compare only those models for which ensembles are computed
# baseline_img = copy(df_images);
# models_img = unique(df_images_ens.modelname);
# ensembles_img = copy(vcat(filter(x -> (x.modelname in models_img), baseline_img), df_images_ens));
# comparison_images_ensemble(
#     baseline_img, 
#     ensembles_img, ("AUC", :auc); 
#     suffix="_only_improve")
# comparison_images_ensemble(
#     baseline_img, 
#     ensembles_img, ("TPR@5", :tpr_5);
#     suffix="_only_improve")
# @info "comparison_images_ensemble_only_improve"

# basic tables when anomaly_class is treated as separate dataset
function basic_tables_images_per_ac(df; suffix="")
    dff = filter(x -> (x.seed == 1), df)
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

# these results are part of supplementary
# df_images_vae = vcat(load(datadir("evaluation/vae_seeds_eval.bson"))[:df], filter(x -> (x.modelname == "vae"), df_images));
# df_images_ganomaly = filter(x -> (x.modelname == "Conv-GANomaly"), df_images)
# function split_seeds!(df)
#     df["modelname"] = string.(df["seed"]) .* df["modelname"]
#     df
# end
# basic_summary_table(split_seeds!(copy(df_images_vae)), prefix="images", suffix="_vae_seed")
# basic_summary_table(split_seeds!(copy(df_images_ganomaly)), prefix="images", suffix="_gano_seed")
# basic_tables_images_per_ac(split_seeds!(copy(df_images_vae)), suffix="_vae_seed")
# basic_tables_images_per_ac(split_seeds!(copy(df_images_ganomaly)), suffix="_gano_seed")

# basic_tables_images_per_ac(copy(df_images))
# basic_tables_images_per_ac(copy(df_images_ens), suffix="_ensembles")
basic_tables_images_per_ac(filter_img_models!(copy(df_images)), suffix="_filter")
basic_tables_images_per_ac(copy(df_images_loi), suffix="_loi")
basic_tables_images_per_ac(filter_img_models!(copy(df_images_ens)), suffix="_ensembles_filter")
@info "basic_tables_images_per_ac"

function plot_knowledge_images_repre(df, models; suffix="", format="pdf")
    filter!(x -> (x.seed == 1), df)
    filter!(x -> (x.modelname in models), df)
    apply_aliases!(df, col="modelname", d=MODEL_MERGE)
    apply_aliases!(df, col="modelname", d=MODEL_ALIAS)
    plot_knowledge_ranks(df; prefix="images", suffix=suffix, format=format)
end

function plot_knowledge_images_type(df; suffix="", format="pdf")
    filter!(x -> (x.seed == 1), df)
    apply_aliases!(df, col="modelname", d=MODEL_MERGE)
    apply_aliases!(df, col="modelname", d=MODEL_TYPE)
    plot_knowledge_ranks(df; prefix="images", suffix=suffix, format=format)
end


representatives=["ocsvm", "aae", "fmgan", "vae_ocsvm"]
plot_knowledge_images_repre(copy(df_images), representatives; format="tex", suffix="_representatives")
# plot_knowledge_images_repre(copy(df_images_ens), representatives; suffix="_representatives_ensembles", format="tex")
@info "plot_knowledge_images_repre"

# these ones are really costly as there are 12 aggregations over all models
# plot_knowledge_images_type(copy(df_images); format="tex") 
# plot_knowledge_images_type(copy(df_images_ens), suffix="_ensembles", format="tex")
# @info "plot_knowledge_images_type"


# this is just specific figure for the main text
function plot_knowledge_combined(df_tab, df_img; format="pdf")
    filter!(x -> (x.seed == 1), df_img);
    apply_aliases!(df_img, col="modelname", d=MODEL_MERGE);
    apply_aliases!(df_img, col="modelname", d=MODEL_TYPE);

    apply_aliases!(df_tab, col="modelname", d=MODEL_MERGE);
    apply_aliases!(df_tab, col="modelname", d=MODEL_TYPE);
    
    for (mn, metric) in [collect(zip(["AUC", "TPR@5"],[:auc, :tpr_5]))[1]]
        val_metric = _prefix_symbol("val", metric)
        tst_metric = _prefix_symbol("tst", metric)

        for (ctype, cnames, criterions) in [collect(zip(
                            ["pat", "pac", "patn"],
                            [PAT_METRICS_NAMES, PAC_METRICS_NAMES, PATN_METRICS_NAMES],
                            [_prefix_symbol.("val", PAT_METRICS), _prefix_symbol.("val", PAC_METRICS), _prefix_symbol.("val", PATN_METRICS)]))[end]]

            function _rank(df, criterions, agg)
                ranks, metric_means = [], []
                for criterion in criterions
                    df_agg = agg(df, criterion)
                    sort!(df_agg, (order(:dataset), order(:modelname)))
                    rt = rank_table(df_agg, tst_metric)
                    push!(ranks, rt[end:end, 2:end])
                    push!(metric_means, mean(Matrix(rt[1:end-3, 2:end]), dims=1))
                end
                vcat(ranks...), vcat(metric_means...)
            end

            function _plot(df_ranks, metric_mean, criterions, cnames, models)
                a = PGFPlots.Axis([PGFPlots.Plots.Linear(
                            1:length(criterions), 
                            df_ranks[:, i]) for (i, m) in enumerate(models)], 
                    ylabel="avg. rnk",
                    style="xtick=$(_pgf_array(1:length(criterions))), 
                        xticklabels=$(_pgf_array(cnames)),
                        width=5cm, height=3cm, scale only axis=true,
                        x tick label style={rotate=50,anchor=east}")
                b = PGFPlots.Axis([PGFPlots.Plots.Linear(
                            1:length(criterions), 
                            metric_mean[:, i], 
                                legendentry=m) for (i, m) in enumerate(models)], 
                    ylabel="avg. $mn",
                    legendStyle = "at={(0.1,1.15)}, anchor=west",
                    style="width=5cm, height=3cm, scale only axis=true, 
                    xtick=$(_pgf_array(1:length(criterions))), 
                    xticklabels={}, legend columns = -1")
                a, b
            end

            extended_criterions = (ctype == "pat") ? criterions[4:end] : criterions
            extended_criterions = vcat(extended_criterions, [val_metric, tst_metric])
            extended_cnames = (ctype == "pat") ? cnames[4:end] : cnames
            extended_cnames = vcat(extended_cnames, ["\$$(mn)_{val}\$", "\$$(mn)_{tst}\$"])

            ranks_tab, metric_means_tab = _rank(df_tab, extended_criterions, aggregate_stats_mean_max)
            p = [1,2,4,5,3] # make sure that flows are last
            models_tab = names(ranks_tab)[p]
            select!(ranks_tab, p)
            metric_means_tab = metric_means_tab[:, p]
            a_tab, b_tab = _plot(ranks_tab, metric_means_tab, extended_criterions, extended_cnames, models_tab)

            extended_criterions = vcat(criterions, [val_metric, tst_metric])
            extended_cnames = vcat(cnames, ["\$$(mn)_{val}\$", "\$$(mn)_{tst}\$"])

            ranks_img, metric_means_img = _rank(df_img, extended_criterions, aggregate_stats_max_mean)
            models_img = names(ranks_img)
            a_img, b_img = _plot(ranks_img, metric_means_img, extended_criterions, extended_cnames, models_img)
            a_img.ylabel = ""
            b_img.ylabel = ""
            for b in b_img.plots
                b.legendentry=nothing
            end

            g = PGFPlots.GroupPlot(2, 2, groupStyle = "vertical sep = 0.5cm, horizontal sep = 1.0cm")
            push!(g, b_tab); push!(g, b_img); 
            push!(g, a_tab); push!(g, a_img);

            filename = "$(projectdir())/paper/figures/combined_knowledge_rank_$(ctype)_$(metric).$(format)"
            PGFPlots.save(filename, g; include_preamble=false)
        end
    end
end

# no filter applied here as we aggregate all the models into groups
# shows the best possible performance
plot_knowledge_combined(copy(df_tabular), copy(df_images); format="tex")
@info "plot_combine_knowledge_type"

@info "----------------- DONE ---------------------"
