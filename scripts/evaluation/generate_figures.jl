using DrWatson
using FileIO, BSON, DataFrames
using PrettyTables
using Statistics
using StatsBase
import PGFPlots

using GenerativeAD.Evaluation: MODEL_ALIAS, DATASET_ALIAS, MODEL_TYPE, apply_aliases!
using GenerativeAD.Evaluation: _prefix_symbol, aggregate_stats_mean_max, aggregate_stats_max_mean
using GenerativeAD.Evaluation: PAT_METRICS, PATN_METRICS, PAC_METRICS, BASE_METRICS, TRAIN_EVAL_TIMES
using GenerativeAD.Evaluation: rank_table, print_rank_table, latex_booktabs, convert_anomaly_class

include("./utils/pgf_boxplot.jl")
include("./utils/ranks.jl")
include("./utils/bayes.jl")

const PAT_METRICS_NAMES = ["\$PR@\\%0.01\$","\$PR@\\%0.1\$","\$PR@\\%1\$","\$PR@\\%5\$","\$PR@\\%10\$","\$PR@\\%20\$"]
const PAC_METRICS_NAMES = ["\$AUC@\\#5\$","\$AUC@\\#10\$","\$AUC@\\#50\$","\$AUC@\\#100\$","\$AUC@\\#500\$","\$AUC@\\#1000\$"]
const PATN_METRICS_NAMES = ["\$PR@\\#5\$","\$PR@\\#10\$","\$PR@\\#50\$","\$PR@\\#100\$","\$PR@\\#500\$","\$PR@\\#1000\$"]

DOWNSAMPLE = Dict(
    zip(["ocsvm", "MAF", "RealNVP"], [100, 100, 100]))

function _tabular_filter!(df)
    # filter out ocsvm_nu
    filter!(x -> (x.modelname != "ocsvm_nu"), df)

    # filter autoencoders others than "aae_full", "wae_full" "vae_full"
    filter!(x -> ~(x.modelname in (["aae", "aae_vamp", "vae", "vae_simple", "wae", "wae_vamp"])), df)           

    # filter only default parameters of "ocsvm" - rbf + nu = 0.5, tune gamma
    filter!(x -> (x.modelname != "ocsvm_rbf") || occursin("nu=0.5", x.parameters), df)
    df
end

function _filter_autoencoders!(df)
    autoencoders = Set([k for (k,v) in MODEL_TYPE if v == "autoencoders"])
    filter!(x -> x.modelname in autoencoders, df)
end

function _filter_ensembles!(df)
    filter!(x -> occursin("ignore_nan=true_method=mean", x.parameters), df)
end


df_tabular = load(datadir("evaluation/tabular_eval.bson"))[:df];
df_tabular_clean = load(datadir("evaluation/tabular_clean_val_final_eval.bson"))[:df];
df_tabular_autoencoders = _filter_autoencoders!(copy(df_tabular));
# with autoencoders separate, it is now safe to just filter out models that do not get into the big comparison
_tabular_filter!(df_tabular);

df_tabular_bayes = load(datadir("evaluation_bayes/tabular_eval.bson"))[:df];
bayes_models = Set(unique(df_tabular_bayes.modelname))
df_tabular_bayes_outerjoin = combine_bayes(df_tabular, df_tabular_bayes; outer=true);
df_tabular_bayes_innerjoin = filter(x -> x.modelname in bayes_models, df_tabular_bayes_outerjoin)


df_tabular_ens = load(datadir("evaluation_ensembles/tabular_eval.bson"))[:df];
_filter_ensembles!(df_tabular_ens)

@info "Loaded results from tabular evaluation."

# works for both image and tabular data
function basic_summary_table(df; suffix="", prefix="", downsample=Dict{String, Int}())
    for metric in [:auc, :tpr_5]
        val_metric = _prefix_symbol("val", metric)
        tst_metric = _prefix_symbol("tst", metric)    

        agg_names = ["maxmean", "meanmax"]
        agg_funct = [aggregate_stats_max_mean, aggregate_stats_mean_max]
        
        for (name, agg) in zip(agg_names, agg_funct)
            _, rt = sorted_rank(df, agg, val_metric, tst_metric, downsample)

            rt[end-2, 1] = "\$\\sigma_1\$"
            rt[end-1, 1] = "\$\\sigma_{10}\$"
            rt[end, 1] = "rnk"

            file = "$(projectdir())/paper/tables/$(prefix)_$(metric)_$(metric)_$(name)$(suffix).tex"
            open(file, "w") do io
                print_rank_table(io, rt; backend=:tex)
            end
        end
    end
end

function basic_summary_table_tabular(df; suffix="", downsample=Dict{String, Int}())
    basic_summary_table(df; prefix="tabular", suffix=suffix, downsample=downsample)
end

## one table with `ocsvm_rbf` and the other with properly tuned `ocsvm`
basic_summary_table_tabular(filter(x->x.modelname != "ocsvm_rbf", df_tabular), suffix="", downsample=DOWNSAMPLE)
basic_summary_table_tabular(filter(x->x.modelname != "ocsvm", df_tabular), suffix="_orbf", downsample=DOWNSAMPLE)

## default/clean parameters table
basic_summary_table_tabular(filter(x->x.modelname != "ocsvm_rbf", df_tabular_clean), suffix="_clean_default")

## bayes results combined with initial random samples and other methods that were not optimized that way
basic_summary_table_tabular(filter(x->x.modelname != "ocsvm_rbf", df_tabular_bayes_outerjoin), suffix="_bayes_outerjoin")

## bayes results combined with initial random samples
basic_summary_table_tabular(df_tabular_bayes_innerjoin, suffix="_bayes_innerjoin")

basic_summary_table_tabular(df_tabular_ens, suffix="_ensembles")
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
                file = "$(projectdir())/paper/figures/$(prefix)_knowledge_rank_$(ctype)_$(metric)_$(name)$(suffix).$(format)"
                PGFPlots.save(file, g; include_preamble=false)
            end
        end
    end
end

function plot_knowledge_tabular_repre(df, models; suffix="", format="pdf", downsample=Dict{String, Int}())
    filter!(x -> (x.modelname in models), df)
    apply_aliases!(df, col="modelname", d=MODEL_ALIAS)
    plot_knowledge_ranks(df; prefix="tabular", suffix=suffix, format=format, downsample=downsample)
end

function plot_knowledge_tabular_type(df; suffix="", format="pdf")
    apply_aliases!(df, col="modelname", d=MODEL_TYPE)
    plot_knowledge_ranks(df; prefix="tabular", suffix=suffix, format=format)
end


representatives=["ocsvm", "wae", "MAF", "fmgan", "vae_ocsvm"]
plot_knowledge_tabular_repre(copy(df_tabular), representatives; suffix="_representatives", format="tex", downsample=DOWNSAMPLE)
@info "plot_knowledge_tabular_repre"

# these ones are really costly as there are 12 aggregations over all models
# plot_knowledge_tabular_type(copy(df_tabular); format="tex") 
@info "plot_knowledge_tabular_type"

# how the choice of creation criterion and size of an ensembles affect results
function ensemble_sensitivity(df, df_ensemble; prefix="tabular", suffix="", downsample=Dict{String, Int}())
    for metric in [:auc, :tpr_5]
        val_metric = _prefix_symbol("val", metric)
        tst_metric = _prefix_symbol("tst", metric)    

        ranks = []
        # computing baseline
        _, rt = sorted_rank(df, aggregate_stats_max_mean, val_metric, tst_metric, downsample)
        models = names(rt)[2:end]
        rt["criterion-size"] = "baseline"
        select!(rt, vcat(["criterion-size"], models))
        push!(ranks, rt[end:end, :])

        # computing different ensembles
        for (cn, c) in zip(["AUC", "TPR@5"], [:val_auc, :val_tpr_5])
            for s in [5, 10]
                dff = filter(x -> startswith(x.parameters, "criterion=$(c)") && endswith(x.parameters, "_size=$(s)"), df_ensemble)
                _, rt_ensemble = sorted_rank(dff, aggregate_stats_max_mean, val_metric, tst_metric, downsample)
                models = names(rt_ensemble)[2:end]
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

        file = "$(projectdir())/paper/tables/$(prefix)_ensemble_size_$(metric)$(suffix).tex"
        open(file, "w") do io
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

ensemble_sensitivity(df_tabular, df_tabular_ens; prefix="tabular", downsample=DOWNSAMPLE)
@info "ensemble_sensitivity_tabular"

# only meanmax as we don't have bayes for images
function bayes_sensitivity(df, df_bayes; prefix="tabular", suffix="", downsample=Dict{String, Int}())
    ranks = []
    
    # differences in performance in both metrics
    for (mn, metric) in zip(["AUC", "TPR@5"],[:auc, :tpr_5])
        val_metric = _prefix_symbol("val", metric)
        tst_metric = _prefix_symbol("tst", metric) 
        
        # computing baseline
        models, rt = sorted_rank(df, aggregate_stats_mean_max, val_metric, tst_metric, downsample)
        rt["exp"] = "random-$(mn)"
        select!(rt, vcat(["exp"], models))
        push!(ranks, rt[end:end, :])
                
        models, rt_bayes = sorted_rank(df_bayes, aggregate_stats_mean_max, val_metric, tst_metric)
        rt_bayes["exp"] = "bayes-$(mn)"
        select!(rt_bayes, vcat(["exp"], models))
        
        metric_dif = Matrix(rt_bayes[1:end-3, 2:end]) - Matrix(rt[1:end-3, 2:end])
        num_inc = sum(metric_dif .> 0, dims=1)
        num_dec = sum(metric_dif .< 0, dims=1)
        mean_dif = round.(mean(metric_dif, dims=1), digits=2)
        
        rank_dif = Vector(rt_bayes[end, 2:end]) - Vector(rt[end, 2:end])

        push!(rt_bayes, ["rank. change", rank_dif...])
        push!(rt_bayes, ["avg. change", mean_dif...])
        push!(rt_bayes, ["num. inc", num_inc...])
        push!(rt_bayes, ["num. dec", num_dec...])
        push!(ranks, rt_bayes[end-4:end, :])
    end
    df_ranks = reduce(vcat, ranks)
    # first six rows correspond to AUC and the rest to TPR
    # 1. random ranks          :: minimum
    # 2. bayes  ranks          :: minimum
    # 3. bayes - random ranks  :: minimum
    # 4. bayes - random metric :: maximum
    # 5. bayes > random metric :: maximum
    # 6. bayes < random metric :: maximum

    min_rows = [1,2,3,7,8,9]
    max_rows = setdiff(collect(1:12), min_rows)
    hl_best_rank = LatexHighlighter(
                    (data, i, j) -> (i in min_rows) && (data[i,j] == minimum(df_ranks[i, 2:end])),
                    ["color{red}","textbf"])

    hl_best_dif = LatexHighlighter(
                    (data, i, j) -> (i in max_rows) && (data[i,j] == maximum(df_ranks[i, 2:end])),
                    ["color{blue}","textbf"])

    f_float = (v, i, j) -> (i == 4 || i == 10) ? ft_printf("%.2f")(v,i,j) : (i in [5,6,11,12]) ? ft_printf("%.0f")(v,i,j) : ft_printf("%.1f")(v,i,j)

    file = "$(projectdir())/paper/tables/$(prefix)_bayes_comp$(suffix).tex"
    open(file, "w") do io
        pretty_table(
            io, df_ranks,
            backend=:latex,
            formatters=f_float,
            highlighters=(hl_best_rank, hl_best_dif),
            nosubheader=true,
            tf=latex_booktabs)
    end
end

bayes_sensitivity(
    filter(x->x.modelname != "ocsvm_rbf", df_tabular), 
    filter(x->x.modelname != "ocsvm_rbf", df_tabular_bayes_outerjoin); prefix="tabular", suffix="", downsample=DOWNSAMPLE)
@info "bayes_sensitivity_tabular"

# training time rank vs avg rank
function plot_fiteval_time(df; time_col=:fit_t, suffix="", prefix="tabular", format="pdf", downsample=Dict{String,Int}())
    
    df["model_type"] = copy(df["modelname"])
    apply_aliases!(df, col="model_type", d=MODEL_TYPE)
    apply_aliases!(df, col="modelname", d=MODEL_ALIAS)
    
    # add first stage time for encoding and training of encoding
    df["fit_t"] .+= df["fs_fit_t"] .+ df["fs_fit_t"]
    # add all eval times together
    df["total_eval_t"] = df["tr_eval_t"] .+ df["tst_eval_t"] .+ df["val_eval_t"]

    # compute ranks from averaged values over each dataset
    # should reduce the noise in the results
    agg_cols = vcat(Symbol.(TRAIN_EVAL_TIMES), :total_eval_t)
    df_time_avg = combine(groupby(df, [:dataset, :modelname]), agg_cols .=> mean .=> agg_cols)
    df_time_avg[_prefix_symbol(time_col, "top_10_std")] = 0.0   # add dummy column
    df_time_avg[_prefix_symbol(time_col, "std")] = 0.0          # add dummy column
    df_time_avg[time_col] .= -df_time_avg[time_col]
    rtt = rank_table(df_time_avg, time_col)

    for (mn, metric) in zip(["AUC", "TPR@5"],[:auc, :tpr_5])
        val_metric = _prefix_symbol("val", metric)
        tst_metric = _prefix_symbol("tst", metric)

        for (name, agg) in zip(
                    ["maxmean", "meanmax"], 
                    [aggregate_stats_max_mean, aggregate_stats_mean_max])

            df_agg = agg(df, val_metric; downsample=downsample)
            rt = rank_table(df_agg, tst_metric)
            
            models = names(rt)[2:end]
            x = Vector(rt[end, 2:end])
            y = Vector(rtt[end, 2:end])
            # labels cannot be shifted uniformly
            a = PGFPlots.Axis([
                    PGFPlots.Plots.Scatter(x, y),
                    [PGFPlots.Plots.Node(m, xx + 0.5, yy + 0.7) for (m, xx, yy) in zip(models, x, y)]...],
                    xlabel="avg. rnk",
                    ylabel="avg. time rnk")
            file = "$(projectdir())/paper/figures/$(prefix)_$(time_col)_vs_$(metric)_$(name)$(suffix).$(format)"
            PGFPlots.save(file, a; include_preamble=false)
        end
    end
end

plot_fiteval_time(df_tabular; time_col=:fit_t, format="tex", downsample=DOWNSAMPLE)
plot_fiteval_time(df_tabular; time_col=:total_eval_t, format="tex", downsample=DOWNSAMPLE)
@info "plot_fiteval_time_tabular"

# show basic table only for autoencoders
function tabular_autoencoders_investigation(df; split_vamp=false, suffix="")
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
            
            file = "$(projectdir())/paper/tables/tabular_ae_only_$(metric)_$(metric)_$(name)$(suffix).tex"
            open(file, "w") do io
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
            file = "$(projectdir())/paper/figures/tabular_ae_only_box_$(metric)_$(name)$(suffix).tex"
            open(file, "w") do f
                write(f, a)
            end
        end
    end
end

tabular_autoencoders_investigation(df_tabular_autoencoders; split_vamp=true, suffix="_split_vamp")
@info "tabular_autoencoders_investigation"

# does crossvalidation matters?
function per_seed_ranks_tabular(df; suffix="", downsample=Dict{String, Int}())
    for metric in [:auc, :tpr_5]
        val_metric = _prefix_symbol("val", metric)
        tst_metric = _prefix_symbol("tst", metric)    

        ranks = []
        for (name, agg) in zip(
                ["maxmean", "meanmax"], 
                [aggregate_stats_max_mean, aggregate_stats_mean_max])
            models, rt = sorted_rank(df, agg, val_metric, tst_metric, downsample)
            rt["exp"] = name
            select!(rt, vcat(["exp"], models))
            push!(ranks, rt[end:end,:])
        end

        for seed in 1:5
            dff = filter(x -> (x.seed == seed), df)
            # silence the warnings for insufficient number of seeds by setting verbose false
            models, rt = sorted_rank(dff, aggregate_stats_max_mean, val_metric, tst_metric, downsample; verbose=false)
            rt["exp"] = "seed=$seed"
            select!(rt, vcat(["exp"], models))
            push!(ranks, rt[end:end,:])
        end

        df_ranks = reduce(vcat, ranks)
        hl_best_rank = LatexHighlighter(
                        (data, i, j) -> (data[i,j] == minimum(df_ranks[i, 2:end])),
                        ["color{red}","textbf"])
    
        file = "$(projectdir())/paper/tables/tabular_per_seed_ranks_$(metric)$(suffix).tex"
        open(file, "w") do io
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

per_seed_ranks_tabular(filter(x->x.modelname != "ocsvm_rbf", df_tabular), suffix="", downsample=DOWNSAMPLE)
per_seed_ranks_tabular(filter(x->x.modelname != "ocsvm", df_tabular), suffix="_orbf", downsample=DOWNSAMPLE)
@info "per_seed_ranks_tabular"

# metric sensitivity
# should be both for image and tabular data
function metric_sensitivity(df; prefix="tabular", suffix="", downsample=Dict{String, Int}())
    ranks = []
    
    # compute rank table for each metric
    for (mn, metric) in zip(["AUC", "TPR@5"],[:auc, :tpr_5])
        val_metric = _prefix_symbol("val", metric)
        tst_metric = _prefix_symbol("tst", metric) 
        
        models, rt = sorted_rank(df, aggregate_stats_auto, val_metric, tst_metric, downsample)
        rt["metric"] = "$(mn)"
        select!(rt, vcat(["metric"], models))
        push!(ranks, rt[end:end, :])
    end
    
    # each row contains average ranks for each metric
    df_ranks = reduce(vcat, ranks)
    rank_dif = Vector(df_ranks[2, 2:end]) .- Vector(df_ranks[1, 2:end])
    push!(df_ranks, ["rank. change", rank_dif...])

    min_rows = [1,2,3]
    max_rows = [3]
    hl_best_rank = LatexHighlighter(
                    (data, i, j) -> (i in min_rows) && (data[i,j] == minimum(df_ranks[i, 2:end])),
                    ["color{red}","textbf"])

    hl_best_dif = LatexHighlighter(
                    (data, i, j) -> (i in max_rows) && (data[i,j] == maximum(df_ranks[i, 2:end])),
                    ["color{blue}","textbf"])

    file = "$(projectdir())/paper/tables/$(prefix)_metric_comp$(suffix).tex"
    open(file, "w") do io
        pretty_table(
            io, df_ranks,
            backend=:latex,
            formatters=ft_printf("%.1f"),
            highlighters=(hl_best_rank, hl_best_dif),
            nosubheader=true,
            tf=latex_booktabs)
    end
end

metric_sensitivity(filter(x->x.modelname != "ocsvm_rbf", df_tabular), suffix="", downsample=DOWNSAMPLE)
metric_sensitivity(filter(x->x.modelname != "ocsvm", df_tabular), suffix="_orbf", downsample=DOWNSAMPLE)
@info "metric_sensitivity"


######################################################################################
#######################               IMAGES                ##########################
######################################################################################
df_images_loo = load(datadir("evaluation/images_leave-one-out_eval.bson"))[:df];
df_images_loi = load(datadir("evaluation/images_leave-one-in_eval.bson"))[:df];
df_images_loo_clean = load(datadir("evaluation/images_leave-one-out_clean_val_final_eval.bson"))[:df];
df_images_loi_clean = load(datadir("evaluation/images_leave-one-in_clean_val_final_eval.bson"))[:df];
# df_images_ens_loo = load(datadir("evaluation_ensembles/images_eval.bson"))[:df];

df_images_mnistc = load(datadir("evaluation/images_mnistc_eval.bson"))[:df];
df_images_mvtec = load(datadir("evaluation/images_mvtec_eval.bson"))[:df];
select!(df_images_mnistc, Not(:anomaly_class)); # mnistc contains anomaly_class but mvtec does not
df_images_single = vcat(df_images_mnistc, df_images_mvtec);
df_images_single[:anomaly_class] = -1; # add dummy anomaly_class to indicate that we have single class anomaly dataset

# there will be also the clean versions
df_images_mnistc_clean = load(datadir("evaluation/images_mnistc_clean_val_final_eval.bson"))[:df];
df_images_mvtec_clean = load(datadir("evaluation/images_mvtec_clean_val_final_eval.bson"))[:df];

### do we have all that we need ?
df_images_loi_clean.modelname |> countmap
df_images_mnistc_clean.modelname |> countmap
df_images_mvtec_clean.modelname |> countmap

df_images_loi.modelname |> countmap
df_images_mnistc.modelname |> countmap
df_images_mvtec.modelname |> countmap
###

select!(df_images_mnistc_clean, Not(:anomaly_class));
df_images_single_clean = vcat(df_images_mnistc, df_images_mvtec_clean);
df_images_single_clean[:anomaly_class] = -1; # this is needed when concatenating with multiclass datasets
@info "Loaded results from images"

function _filter_image_multi!(df)
    filter!(x -> x.seed == 1, df)
    filter!(x -> ~(x.modelname in ["aae_ocsvm", "fAnoGAN-GP"]), df)
end

_filter_image_multi!(df_images_loo);
_filter_image_multi!(df_images_loi);
_filter_image_multi!(df_images_loo_clean);
_filter_image_multi!(df_images_loi_clean);
@info "Applied basic filters"

basic_summary_table(df_images_loo, prefix="images_multi", suffix="_loo")
basic_summary_table(df_images_loi, prefix="images_multi", suffix="_loi")
basic_summary_table(df_images_loo_clean, prefix="images_multi", suffix="_loo_clean_default")
basic_summary_table(df_images_loi_clean, prefix="images_multi", suffix="_loi_clean_default")

basic_summary_table(df_images_single; prefix="images_single", suffix="")
@info "basic_summary_table_images separated"


df_images = vcat(df_images_loi, df_images_single);
df_images_clean = vcat(df_images_loi_clean, df_images_single_clean);

SEMANTIC_IMAGE_ANOMALIES = Set(["CIFAR10", "SVHN2", "MVTec-AD_grid", "MVTec-AD_transistor", "MVTec-AD_wood"])
# SEMANTIC_IMAGE_ANOMALIES = Set(["CIFAR10", "SVHN2", "MVTec-AD_grid", "MVTec-AD_transistor", "MVTec-AD_wood", "MNIST-C_rotate" , "MNIST-C_scale", "MNIST-C_shear", "MNIST-C_translate"])

# splits single and multi class image datasets into "statistic" and "semantic" anomalies
_split_image_datasets(df) = (
            filter(x -> x.dataset in SEMANTIC_IMAGE_ANOMALIES, df), 
            filter(x -> ~(x.dataset in SEMANTIC_IMAGE_ANOMALIES), df)
        )

df_images_semantic, df_images_stat = _split_image_datasets(df_images);
df_images_semantic_clean, df_images_stat_clean =  _split_image_datasets(df_images_clean);

# works for both image and tabular data assuming that tabular has anomaly_class column = -1
function basic_summary_table_autoagg(df; suffix="", prefix="", downsample=Dict{String, Int}())
    for metric in [:auc, :tpr_5]
        val_metric = _prefix_symbol("val", metric)
        tst_metric = _prefix_symbol("tst", metric)    

        _, rt = sorted_rank(df, aggregate_stats_auto, val_metric, tst_metric, downsample)

        rt[end-2, 1] = "\$\\sigma_1\$"
        rt[end-1, 1] = "\$\\sigma_{10}\$"
        rt[end, 1] = "rnk"

        file = "$(projectdir())/paper/tables/$(prefix)_$(metric)_$(metric)_autoagg$(suffix).tex"
        open(file, "w") do io
            print_rank_table(io, rt; backend=:tex)
        end
    end
end

basic_summary_table_autoagg(df_images_semantic, prefix="images_semantic", suffix="")
basic_summary_table_autoagg(df_images_stat, prefix="images_stat", suffix="")
@info "basic_summary_table_images stat/semantic"

basic_summary_table(df_images_semantic, prefix="images_semantic", suffix="_clean_default")
basic_summary_table(df_images_stat, prefix="images_stat", suffix="_clean_default")
@info "basic_summary_table_images stat/semantic clean"

function comparison_images_ensemble(df, df_ensemble, tm=("AUC", :auc); suffix="")
    filter!(x -> (x.seed == 1), df)
    filter!(x -> (x.seed == 1), df_ensemble)
    comparison_ensemble(df, df_ensemble, tm; prefix="images", suffix=suffix)
end

function ensemble_sensitivity_images(df, df_ensemble)
    ensemble_sensitivity(df, df_ensemble; prefix="images")
end

ensemble_sensitivity_images(
        filter_img_models!(copy(df_images)),
        _filter_ensembles!(filter_img_models!(copy(df_images_ens))))

@info "ensemble_sensitivity_images"

# basic tables when anomaly_class is treated as separate dataset
function basic_tables_images_per_ac(df; suffix="")
    dff = filter(x -> (x.seed == 1), df)
    apply_aliases!(dff, col="dataset", d=DATASET_ALIAS)
    for d in unique(dff.dataset)
        mask = (dff.dataset .== d)
        dff[mask, :dataset] .= dff[mask, :dataset] .* ":" .* convert_anomaly_class.(dff[mask, :anomaly_class], d)
    end
    select!(dff, Not(:anomaly_class))

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

        file = "$(projectdir())/paper/tables/images_per_ac_$(metric)_$(metric)$(suffix).tex"
        open(file, "w") do io
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

basic_tables_images_per_ac(filter_img_models!(copy(df_images)), suffix="_filter")
basic_tables_images_per_ac(copy(df_images_loi), suffix="_loi")

@info "basic_tables_images_per_ac"

function plot_knowledge_images_repre(df, models; suffix="", format="pdf")
    filter!(x -> (x.modelname in models), df)
    apply_aliases!(df, col="modelname", d=MODEL_ALIAS)
    plot_knowledge_ranks(df; prefix="images", suffix=suffix, format=format)
end

function plot_knowledge_images_type(df; suffix="", format="pdf")
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

# this function combines all the image datasets and applies specific agg to each of them
# this can be rewritten using the autoagg function
function plot_fiteval_time_image_custom(dfloo, dfloi, dfc; time_col=:fit_t, suffix="", format="pdf")
    
    df_times = map([dfloo, dfloi, dfc]) do df
        # add first stage time for encoding and training of encoding
        df["fit_t"] .+= df["fs_fit_t"] .+ df["fs_fit_t"]
        # add all eval times together
        df["total_eval_t"] = df["tr_eval_t"] .+ df["tst_eval_t"] .+ df["val_eval_t"]

        # compute ranks from averaged values over each dataset
        # should reduce the noise in the results
        agg_cols = vcat(Symbol.(TRAIN_EVAL_TIMES), :total_eval_t)
        df_time_avg = combine(groupby(df, [:dataset, :modelname]), agg_cols .=> mean .=> agg_cols)
        df_time_avg[_prefix_symbol(time_col, "top_10_std")] = 0.0   # add dummy column
        df_time_avg[_prefix_symbol(time_col, "std")] = 0.0          # add dummy column
        df_time_avg[time_col] .= -df_time_avg[time_col]
        apply_aliases!(df_time_avg, col="modelname", d=MODEL_ALIAS)
        
        df_time_avg
    end

    # distinguish LOI and LOO datasets
    df_times[2][:dataset] .= df_times[2][:dataset] .* "_loi"

    df_time = reduce(vcat, df_times)
    # compute ranks accross all datasets
    rtt = rank_table(df_time, time_col)
    models = names(rtt)[2:end]
    
    for (mn, metric) in zip(["AUC", "TPR@5"],[:auc, :tpr_5])
        val_metric = _prefix_symbol("val", metric)
        tst_metric = _prefix_symbol("tst", metric)

        df_aggs = map(zip([dfloo, dfloi, dfc], [aggregate_stats_max_mean, aggregate_stats_max_mean, aggregate_stats_mean_max])) do (df, agg)
            df_agg = agg(df, val_metric)
        end
        # distinguish LOI and LOO datasets
        df_aggs[2][:dataset] .= df_aggs[2][:dataset] .* "_loi"
        # remove columns not present in maxmean aggregation
        select!(df_aggs[3], Not([:phash, :psamples, :parameters, :dsamples, :dsamples_valid]))

        df_agg = reduce(vcat, df_aggs)

        rt = rank_table(df_agg, tst_metric)

        x = Vector(rt[end, 2:end])
        y = Vector(rtt[end, 2:end])

        a = PGFPlots.Axis([
                PGFPlots.Plots.Scatter(x, y),
                [PGFPlots.Plots.Node(m, xx + 0.5, yy + 0.7) for (m, xx, yy) in zip(models, x, y)]...],
                xlabel="avg. rnk",
                ylabel="avg. time rnk")
        file = "$(projectdir())/paper/figures/images_$(time_col)_vs_$(metric)_data_combined.$(format)"
        PGFPlots.save(file, a; include_preamble=false)
    end
end

# plot_fiteval_time(df_images)); prefix="images", time_col=:fit_t, format="tex", suffix="_filter")
# plot_fiteval_time(df_images)); prefix="images", time_col=:total_eval_t, format="tex", suffix="_filter")
# plot_fiteval_time(copy(df_images_loi); prefix="images", time_col=:fit_t, format="tex", suffix="_loi")
# plot_fiteval_time(copy(df_images_loi); prefix="images", time_col=:total_eval_t, format="tex", suffix="_loi")
# plot_fiteval_time(filter_class_datasets!(copy(df_images_class)); prefix="images", time_col=:fit_t, format="tex", suffix="_class")
# plot_fiteval_time(filter_class_datasets!(copy(df_images_class)); prefix="images", time_col=:total_eval_t, format="tex", suffix="_class")
# @info "plot_fiteval_time_images"


plot_fiteval_time_image_custom(filter_img_models!(copy(df_images)), copy(df_images_loi), filter_class_datasets!(copy(df_images_class)); time_col=:fit_t, format="tex")
plot_fiteval_time_image_custom(filter_img_models!(copy(df_images)), copy(df_images_loi), filter_class_datasets!(copy(df_images_class)); time_col=:total_eval_t, format="tex")
@info "plot_fiteval_time_image_custom"

# this is just specific figure for the main text
# grouptype = true  - renames models according to model type
# grouptype = false - assumes that the results are filtered to a few models/representatives
function plot_knowledge_combined(df_tab, df_img_stat, df_img_sem; grouptype=true, format="pdf", suffix="")
    df_tab, df_tab_clean = df_tab;
    df_img_stat, df_img_stat_clean = df_img_stat;
    df_img_sem, df_img_sem_clean = df_img_sem;

    # tabulars are single class anomaly datasets
    df_tab[:anomaly_class] = -1
    df_img_stat_clean = select(df_img_stat_clean, Not(:anomaly_class))
    df_img_sem_clean = select(df_img_sem_clean, Not(:anomaly_class))

    # rename models to their corresponding type
    if grouptype
        for df in [df_tab, df_tab_clean, df_img_stat, df_img_stat_clean, df_img_sem, df_img_sem_clean]
            apply_aliases!(df, col="modelname", d=MODEL_TYPE);
        end
    end

    for (mn, metric) in [collect(zip(["AUC", "TPR@5"],[:auc, :tpr_5]))[1]]
        val_metric = _prefix_symbol("val", metric)
        tst_metric = _prefix_symbol("tst", metric)

        # ctype = "patn"
        cnames = PATN_METRICS_NAMES
        criterions = _prefix_symbol.("val", PATN_METRICS)

        function _incremental_rank(df, criterions, agg)
            ranks, metric_means = [], []
            for criterion in criterions
                df_agg = agg(df, criterion)
                sort!(df_agg, [:dataset, :modelname])
                rt = rank_table(df_agg, tst_metric)
                push!(ranks, rt[end:end, 2:end])
                push!(metric_means, mean(Matrix(rt[1:end-3, 2:end]), dims=1))
            end
            vcat(ranks...), vcat(metric_means...)
        end

        function _plot(df_ranks, metric_mean, cnames, models)
            a = PGFPlots.Axis([PGFPlots.Plots.Linear(
                        1:length(cnames), 
                        df_ranks[:, i]) for (i, m) in enumerate(models)], 
                ylabel="avg. rnk",
                style="xtick=$(_pgf_array(1:length(cnames))), 
                    xticklabels=$(_pgf_array(cnames)),
                    width=5cm, height=3cm, scale only axis=true,
                    x tick label style={rotate=50,anchor=east}")
            b = PGFPlots.Axis([PGFPlots.Plots.Linear(
                        1:length(cnames), 
                        metric_mean[:, i], 
                            legendentry=m) for (i, m) in enumerate(models)], 
                ylabel="avg. $mn",
                legendStyle = "at={(0.1,1.15)}, anchor=west",
                style="width=5cm, height=3cm, scale only axis=true, 
                xtick=$(_pgf_array(1:length(cnames))), 
                xticklabels={}, legend columns = -1")
            a, b
        end
        
        extended_criterions = vcat(criterions, [val_metric])
        extended_cnames = vcat(["clean"], vcat(cnames, ["\$$(mn)_{val}\$"]))
        
        ab_plots = map(enumerate([(df_tab, df_tab_clean), (df_img_stat, df_img_stat_clean), (df_img_sem, df_img_sem_clean)])) do (i, (df, df_clean))
            ranks_clean, metric_means_clean = _incremental_rank(df_clean, [val_metric], aggregate_stats_max_mean)
            ranks, metric_means = _incremental_rank(df, extended_criterions, aggregate_stats_auto)

            ranks, metric_means = vcat(ranks_clean, ranks), vcat(metric_means_clean, metric_means)
            
            # reorder table on tabular data as there is additional class of models (flows)
            # one can do this manually at the end
            if i == 1 && grouptype
                p = [1,2,4,5,3] # make sure that flows are last
                models = names(ranks)[p]
                select!(ranks, p)
                metric_means = metric_means[:, p]
                a, b = _plot(ranks, metric_means, extended_cnames, models)
            else
                models = names(ranks)
                a, b = _plot(ranks, metric_means, extended_cnames, models)
            end
            a, b
        end
        
        a_tab, b_tab = ab_plots[1]
        a_img_stat, b_img_stat = ab_plots[2]
        a_img_sem, b_img_sem = ab_plots[3]
        
        a_img_stat.ylabel = ""
        b_img_stat.ylabel = ""
        a_img_sem.ylabel = ""
        b_img_sem.ylabel = ""
        if grouptype
            for (bst, bsm) in zip(b_img_stat.plots, b_img_sem.plots)
                bst.legendentry=nothing
                bsm.legendentry=nothing
            end
        end

        g = PGFPlots.GroupPlot(3, 2, groupStyle = "vertical sep = 0.5cm, horizontal sep = 1.0cm")
        push!(g, b_tab); push!(g, b_img_stat); push!(g, b_img_sem);
        push!(g, a_tab); push!(g, a_img_stat); push!(g, a_img_sem);

        file = "$(projectdir())/paper/figures/combined_knowledge_rank_$(metric)$(suffix).$(format)"
        PGFPlots.save(file, g; include_preamble=false)
    end
end

### TEMPORARY
# df_tab = df_tabular, df_tabular_clean;
# df_img_stat = df_images_stat, df_images_stat_clean;
# df_img_sem = df_images_semantic, df_images_semantic_clean;
# format = "tex"
###

plot_knowledge_combined(
    (df_tabular, df_tabular_clean), 
    (df_images_stat, df_images_stat_clean), 
    (df_images_semantic, df_images_semantic_clean); format="tex", suffix="_grouptype")

representatives=["ocsvm", "knn", "wae_full", "wae", "RealNVP", "fmgan", "DeepSVDD"]
ff(df) = filter(x -> x.modelname in representatives, df)

df_tab = df_tabular |> ff, df_tabular_clean |> ff;
df_img_stat = df_images_stat |> ff, df_images_stat_clean |> ff;
df_img_sem = df_images_semantic |> ff, df_images_semantic_clean |> ff;

plot_knowledge_combined(df_tab, df_img_stat, df_img_sem; 
                    grouptype=false, format="tex", suffix="_repre")
@info "plot_combine_knowledge_type"

@info "----------------- DONE ---------------------"
