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
df_tabular_bayes_outerjoin = combine_bayes(df_tabular, df_tabular_bayes; outer=true);
bayes_models = Set(unique(df_tabular_bayes.modelname))
df_tabular_bayes_innerjoin = filter(x -> x.modelname in bayes_models, df_tabular_bayes_outerjoin);

df_tabular[:anomaly_class] = -1;
df_tabular_clean[:anomaly_class] = 1; # in order to trigger maxmean during autoagg

df_tabular_ens = load(datadir("evaluation_ensembles/tabular_eval.bson"))[:df];
_filter_ensembles!(df_tabular_ens);

@info "Loaded results from tabular evaluation."

# works for both image and tabular data
function basic_summary_table(df; suffix="", prefix="", downsample=Dict{String, Int}())
    agg_names = ["maxmean", "meanmax"]
    agg_funct = [aggregate_stats_max_mean, aggregate_stats_mean_max]
    for (name, agg) in zip(agg_names, agg_funct)
        for metric in [:auc, :tpr_5]
            val_metric = _prefix_symbol("val", metric)
            tst_metric = _prefix_symbol("tst", metric)    
            
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

## one table with `ocsvm_rbf` and the other with properly tuned `ocsvm`
basic_summary_table(filter(x->x.modelname != "ocsvm_rbf", df_tabular); prefix="tabular", suffix="", downsample=DOWNSAMPLE)
basic_summary_table(filter(x->x.modelname != "ocsvm", df_tabular); prefix="tabular", suffix="_orbf", downsample=DOWNSAMPLE)

## default/clean parameters table
basic_summary_table(filter(x->x.modelname != "ocsvm_rbf", df_tabular_clean); prefix="tabular", suffix="_clean_default")

## bayes results combined with initial random samples and other methods that were not optimized that way
basic_summary_table(filter(x->x.modelname != "ocsvm_rbf", df_tabular_bayes_outerjoin); prefix="tabular", suffix="_bayes_outerjoin")

## bayes results combined with initial random samples
basic_summary_table(df_tabular_bayes_innerjoin; prefix="tabular", suffix="_bayes_innerjoin")

basic_summary_table(df_tabular_ens; prefix="tabular", suffix="_ensembles")
@info "basic_summary_table_tabular"

# how the choice of creation criterion and size of an ensembles affect results
function ensemble_sensitivity(df, df_ensemble; prefix="tabular", suffix="", downsample=Dict{String, Int}())
    for metric in [:auc, :tpr_5]
        val_metric = _prefix_symbol("val", metric)
        tst_metric = _prefix_symbol("tst", metric)    

        ranks = []
        # computing baseline
        models, rt = sorted_rank(df, aggregate_stats_max_mean, val_metric, tst_metric, downsample)
        rt["criterion-size"] = "baseline"
        select!(rt, vcat(["criterion-size"], models))
        push!(ranks, rt[end:end, :])

        # computing different ensembles
        for (cn, c) in zip(["AUC", "TPR@5"], [:val_auc, :val_tpr_5])
            for s in [5, 10]
                dff = filter(x -> startswith(x.parameters, "criterion=$(c)") && endswith(x.parameters, "_size=$(s)"), df_ensemble)
                models, rt_ensemble = sorted_rank(dff, aggregate_stats_max_mean, val_metric, tst_metric, downsample; verbose=false)
                rt_ensemble["criterion-size"] = "$(cn)-$(s)"
                select!(rt_ensemble, vcat(["criterion-size"], models))
                mean_dif = DataFrame([Symbol(model) => round.(mean(rt_ensemble[1:end-3, model] .- rt[1:end-3, model]), digits=2) for model in models])
                mean_dif["criterion-size"] = "$(cn)-$(s)"

                push!(ranks, rt_ensemble[end:end, :])
                push!(ranks, mean_dif)
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

ensemble_sensitivity(filter(x->x.modelname != "ocsvm_rbf", df_tabular), df_tabular_ens; prefix="tabular", downsample=DOWNSAMPLE)
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
# should work for all dataset types as it uses the automatic aggregation
function plot_fiteval_time(df; time_col=:fit_t, suffix="", prefix="tabular", format="pdf", downsample=Dict{String,Int}())
    # add first stage time for encoding and training of encoding
    df["total_fit_t"] = df["fit_t"] .+ df["fs_fit_t"]
    # add all eval times together
    df["total_eval_t"] = df["tr_eval_t"] .+ df["tst_eval_t"] .+ df["val_eval_t"] .+ df["fs_eval_t"]
    
    # compute ranks from averaged values over each dataset
    # should reduce the noise in the results
    agg_cols = vcat(Symbol.(TRAIN_EVAL_TIMES), [:total_eval_t, :total_fit_t])
    df_time_avg = combine(groupby(df, [:dataset, :modelname]), agg_cols .=> mean .=> agg_cols)
    df_time_avg[_prefix_symbol(time_col, "top_10_std")] = 0.0   # add dummy column
    df_time_avg[_prefix_symbol(time_col, "std")] = 0.0          # add dummy column
    df_time_avg[time_col] .= -df_time_avg[time_col]

    # applying this to a copy of a dataframe is prefered
    apply_aliases!(df_time_avg, col="modelname", d=MODEL_RENAME)
	apply_aliases!(df_time_avg, col="modelname", d=MODEL_ALIAS)
	apply_aliases!(df_time_avg, col="dataset", d=DATASET_ALIAS)
    # this is to ensure how the rank_table is sorted
	sort!(df_time_avg, (:dataset, :modelname))

    rtt = rank_table(df_time_avg, time_col)
    for (mn, metric) in zip(["AUC", "TPR@5"],[:auc, :tpr_5])
        val_metric = _prefix_symbol("val", metric)
        tst_metric = _prefix_symbol("tst", metric)

        df_agg = aggregate_stats_auto(df, val_metric; downsample=downsample)
        apply_aliases!(df_agg, col="modelname", d=MODEL_RENAME)
        apply_aliases!(df_agg, col="modelname", d=MODEL_ALIAS)
        apply_aliases!(df_agg, col="dataset", d=DATASET_ALIAS)
                    
        # this is to ensure how the rank_table is sorted
        sort!(df_agg, (:dataset, :modelname))
        
        rt = rank_table(df_agg, tst_metric)
        
        models = names(rt)[2:end]
        x = Vector(rt[end, 2:end])
        y = Vector(rtt[end, 2:end])
        # labels cannot be shifted uniformly
        a = PGFPlots.Axis([
                PGFPlots.Plots.Scatter(x, y),
                [PGFPlots.Plots.Node(m, xx + 0.5, yy + 0.7) for (m, xx, yy) in zip(models, x, y)]...],
                xlabel="avg. rnk",
                ylabel="avg. \$$(String(time_col))\$ rnk")
        file = "$(projectdir())/paper/figures/$(prefix)_$(time_col)_vs_$(metric)$(suffix).$(format)"
        PGFPlots.save(file, a; include_preamble=false)
    end
end

plot_fiteval_time(filter(x->x.modelname != "ocsvm_rbf", df_tabular); time_col=:total_fit_t, format="tex", downsample=DOWNSAMPLE)
plot_fiteval_time(filter(x->x.modelname != "ocsvm_rbf", df_tabular); time_col=:total_eval_t, format="tex", downsample=DOWNSAMPLE)
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

# needs dummy anomaly_class column
metric_sensitivity(filter(x->x.modelname != "ocsvm_rbf", df_tabular), suffix="", downsample=DOWNSAMPLE)
metric_sensitivity(filter(x->x.modelname != "ocsvm_rbf", df_tabular_clean), suffix="_clean_default")
@info "metric_sensitivity"


######################################################################################
#######################               IMAGES                ##########################
######################################################################################
# df_images_loo = load(datadir("evaluation/images_leave-one-out_eval.bson"))[:df];
# df_images_loo_clean = load(datadir("evaluation/images_leave-one-out_clean_val_final_eval.bson"))[:df];
# filter!(x -> x.seed == 1, df_images_loo)
# filter!(x -> ~(x.modelname in ["aae_ocsvm", "fAnoGAN-GP"]), df_images_loo)
df_images_loi = load(datadir("evaluation/images_leave-one-in_eval.bson"))[:df];
df_images_loi_clean = load(datadir("evaluation/images_leave-one-in_clean_val_final_eval.bson"))[:df];
df_images_loi_clean[df_images_loi_clean.fit_t .=== nothing, :fit_t] .= 1.0; # ConvSkipGANomaly is missing the fit_t

df_images_mnistc = load(datadir("evaluation/images_mnistc_eval.bson"))[:df];
filter!(x -> x.dataset != "MNIST-C_zigzag", df_images_mnistc); # filter out MNISTC-C_zigzag which we forgot and could not compute in time
df_images_mvtec = load(datadir("evaluation/images_mvtec_eval.bson"))[:df];
select!(df_images_mnistc, Not(:anomaly_class)); # mnistc contains anomaly_class but mvtec does not
df_images_single = vcat(df_images_mnistc, df_images_mvtec);
df_images_single[:anomaly_class] = -1; # add dummy anomaly_class to indicate that we have single class anomaly dataset

df_images_mnistc_clean = load(datadir("evaluation/images_mnistc_clean_val_final_eval.bson"))[:df];
filter!(x -> x.dataset != "MNIST-C_zigzag", df_images_mnistc_clean); # filter out MNISTC-C_zigzag which we forgot and could not compute in time
df_images_mvtec_clean = load(datadir("evaluation/images_mvtec_clean_val_final_eval.bson"))[:df];

select!(df_images_mnistc_clean, Not(:anomaly_class));
df_images_single_clean = vcat(df_images_mnistc_clean, df_images_mvtec_clean);
# df_images_single_clean[:anomaly_class] = -1; # this is needed when concatenating with multiclass datasets
df_images_single_clean[:anomaly_class] = 1; # this is needed when concatenating with multiclass datasets
@info "Loaded image results"

df_images_loi_ens = load(datadir("evaluation_ensembles/images_leave-one-in_eval.bson"))[:df];
df_images_mnistc_ens = load(datadir("evaluation_ensembles/images_mnistc_eval.bson"))[:df];
df_images_mvtec_ens = load(datadir("evaluation_ensembles/images_mvtec_eval.bson"))[:df];
df_images_mvtec_ens[:anomaly_class] = -1;
_filter_ensembles!(df_images_loi_ens);
_filter_ensembles!(df_images_mnistc_ens);
_filter_ensembles!(df_images_mvtec_ens);

df_images = vcat(df_images_loi, df_images_single);
df_images_clean = vcat(df_images_loi_clean, df_images_single_clean);
df_images_ens = vcat(df_images_loi_ens, df_images_mnistc_ens, df_images_mvtec_ens);

# renaming dataset to dataset:ac
for df in [df_images, df_images_clean, df_images_ens]
    apply_aliases!(df, col="dataset", d=DATASET_ALIAS)
    for d in Set(["cifar10", "svhn2", "mnist", "fmnist"])
        mask = (df.dataset .== d)
        df[mask, :dataset] .= df[mask, :dataset] .* ":" .* convert_anomaly_class.(df[mask, :anomaly_class], d)
        df[mask, :anomaly_class] .= 1 # it has to be > 0, because otherwise we get too many warnings from the aggregate_stats_max_mean
    end
end

@info "Loaded ensemble image results"
#= do we have all that we need ?
df_images_loi.modelname |> countmap
df_images_mnistc.modelname |> countmap
df_images_mvtec.modelname |> countmap

df_images_loi_clean.modelname |> countmap
df_images_mnistc_clean.modelname |> countmap
df_images_mvtec_clean.modelname |> countmap

df_images_loi.dataset |> countmap
df_images_mnistc.dataset |> countmap
df_images_mvtec.dataset |> countmap

df_images_loi_clean.dataset |> countmap
df_images_mnistc_clean.dataset |> countmap
df_images_mvtec_clean.dataset |> countmap
=#

basic_summary_table(df_images_loi, prefix="images_multi", suffix="")
basic_summary_table(df_images_loi_clean, prefix="images_multi", suffix="_clean_default")
basic_summary_table(df_images_single; prefix="images_single", suffix="")
basic_summary_table(df_images_single_clean; prefix="images_single", suffix="_clean_default")
@info "basic_summary_table_images separated"

SEMANTIC_IMAGE_ANOMALIES = Set(["CIFAR10", "SVHN2"])

# splits single and multi class image datasets into "statistic" and "semantic" anomalies
_split_image_datasets(df) = (
            filter(x -> x.dataset in SEMANTIC_IMAGE_ANOMALIES, df), 
            filter(x -> ~(x.dataset in SEMANTIC_IMAGE_ANOMALIES), df)
        )

df_images_semantic, df_images_stat = _split_image_datasets(df_images);
df_images_semantic_clean, df_images_stat_clean =  _split_image_datasets(df_images_clean);
df_images_semantic_ens, df_images_stat_ens = _split_image_datasets(df_images_ens);

basic_summary_table_autoagg(df_images_semantic, prefix="images_semantic", suffix="_per_ac")
basic_summary_table_autoagg(df_images_stat, prefix="images_stat", suffix="_per_ac")
@info "basic_summary_table_images stat/semantic"

basic_summary_table(df_images_semantic_clean, prefix="images_semantic", suffix="_clean_default_per_ac")
basic_summary_table(df_images_stat_clean, prefix="images_stat", suffix="_clean_default_per_ac")
@info "basic_summary_table_images stat/semantic clean"

metric_sensitivity(df_images_semantic, prefix="images_semantic", suffix="_per_ac")
metric_sensitivity(df_images_stat, prefix="images_stat", suffix="_per_ac")

df_images_stat_clean[:anomaly_class] = 1;       # in order to trigger maxmean agg
df_images_semantic_clean[:anomaly_class] = 1;   # in order to trigger maxmean agg
metric_sensitivity(df_images_semantic_clean, prefix="images_semantic", suffix="_clean_default_per_ac")
metric_sensitivity(df_images_stat_clean, prefix="images_stat", suffix="_clean_default_per_ac")
@info "metric_sensitivity"

ensemble_sensitivity(df_images_semantic, df_images_semantic_ens; prefix="images_semantic", suffix="_per_ac")
ensemble_sensitivity(df_images_stat, df_images_stat_ens; prefix="images_stat", suffix="_per_ac")
@info "ensemble_sensitivity"

plot_fiteval_time(df_images; prefix="images", time_col=:total_fit_t, format="tex", suffix="_per_ac")
plot_fiteval_time(df_images; prefix="images", time_col=:total_eval_t, format="tex", suffix="_per_ac")
@info "plot_fiteval_time_images"


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

# this is just specific figure for the main text
# grouptype = true  - renames models according to model type
# grouptype = false - assumes that the results are filtered to a few models/representatives
function plot_knowledge_combined(df_tab, df_img_stat, df_img_sem; grouptype=true, format="pdf", suffix="")
    df_tab, df_tab_clean = df_tab;
    df_img_stat, df_img_stat_clean = df_img_stat;
    df_img_sem, df_img_sem_clean = df_img_sem;

    # tabulars are single class anomaly datasets
    df_tab[:anomaly_class] = -1
    # not include anomaly_class in DataFrame to avoid a wall of warnings about missing anomaly_class from maxmean agg
    # df_img_stat_clean = select(df_img_stat_clean, Not(:anomaly_class))
    # df_img_sem_clean = select(df_img_sem_clean, Not(:anomaly_class))

    # rename models to their corresponding type
    if grouptype
        for df in [df_tab, df_tab_clean, df_img_stat, df_img_stat_clean, df_img_sem, df_img_sem_clean]
            apply_aliases!(df, col="modelname", d=MODEL_TYPE);
        end
    end

    for (mn, metric) in [collect(zip(["AUC", "TPR@5"],[:auc, :tpr_5]))[1]]
        val_metric = _prefix_symbol("val", metric)
        tst_metric = _prefix_symbol("tst", metric)

        function _incremental_rank(df, criterions, agg)
            ranks, metric_means = [], []
            for criterion in criterions
                df_agg = agg(df, criterion)
                apply_aliases!(df_agg, col="modelname", d=MODEL_RENAME)
                apply_aliases!(df_agg, col="modelname", d=MODEL_ALIAS)
                sort!(df_agg, [:dataset, :modelname])
                rt = rank_table(df_agg, tst_metric)
                # _meanfinite(a) = mean([x for x in a if !isnan(x)])
                # mm = DataFrame([Symbol(model) => _meanfinite(rt[1:end-3, model]) for model in names(rt)[2:end]])
                mm = DataFrame([Symbol(model) => mean(rt[1:end-3, model]) for model in names(rt)[2:end]])
                # @info("", criterion, rt, mm)
                push!(ranks, rt[end:end, 2:end])
                push!(metric_means, mm)
            end
            vcat(ranks...), vcat(metric_means...)
        end
        
        function _plot(df_ranks, metric_mean, cnames, models, title)
            a = PGFPlots.Axis([PGFPlots.Plots.Linear(
                        1:length(cnames), 
                        df_ranks[:, m]) for m in models], 
                ylabel="avg. rnk",
                title=title,
                style="xtick=$(_pgf_array(1:length(cnames))), 
                    xticklabels=$(_pgf_array(cnames)),
                    width=5cm, height=3cm, scale only axis=true,
                    x tick label style={rotate=50,anchor=east},
                    title style={at={(current bounding box.south)}, anchor=west}")
            b = PGFPlots.Axis([PGFPlots.Plots.Linear(
                        1:length(cnames), 
                        metric_mean[:, m], 
                            legendentry=m) for m in models],
                ylabel="avg. $mn",
                legendStyle = "at={(0.3,1.30)}, anchor=west",
                style="width=5cm, height=3cm, scale only axis=true, 
                xtick=$(_pgf_array(1:length(cnames))), 
                xticklabels={}")
            a, b
        end
        
        for (ctype, cnames, criterions) in collect(zip(
            ["pat", "pac", "patn"],
            [PAT_METRICS_NAMES, PAC_METRICS_NAMES, PATN_METRICS_NAMES],
            [_prefix_symbol.("val", PAT_METRICS), _prefix_symbol.("val", PAC_METRICS), _prefix_symbol.("val", PATN_METRICS)]))[1:1]
        
            extended_criterions = vcat(criterions, [val_metric])
            extended_cnames = vcat(["clean"], vcat(cnames, ["\$$(mn)_{val}\$"]))
            titles = ["(tab)", "(stat)", "(semantic)"]
            
            ab_plots = map(enumerate([(df_tab, df_tab_clean), (df_img_stat, df_img_stat_clean), (df_img_sem, df_img_sem_clean)])) do (i, (df, df_clean))
                ranks_clean, metric_means_clean = _incremental_rank(df_clean, [val_metric], aggregate_stats_max_mean)
                ranks_inc, metric_means_inc = _incremental_rank(df, extended_criterions, aggregate_stats_auto)
                
                ranks_all, metric_means_all = vcat(ranks_clean, ranks_inc; cols=:intersect), vcat(metric_means_clean, metric_means_inc; cols=:intersect)
                # @info("", ranks_clean, ranks_inc)
                # @info("", metric_means_clean, metric_means_inc)

                # reorder table on tabular data as there is additional class of models (flows)
                # one can do this manually at the end
                if i == 1 && grouptype
                    p = [1,2,4,5,3] # make sure that flows are last
                    models = names(ranks_all)[p]
                    select!(ranks_all, p)
                    select!(metric_means_all, p)
                    a, b = _plot(ranks_all, metric_means_all, extended_cnames, models, titles[i])
                else
                    models = names(ranks_all)
                    a, b = _plot(ranks_all, metric_means_all, extended_cnames, models, titles[i])
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

            file = "$(projectdir())/paper/figures/combined_knowledge_rank_$(ctype)_$(metric)$(suffix).$(format)"
            PGFPlots.save(file, g; include_preamble=false)
        end
    end
end

plot_knowledge_combined(
    (df_tabular, df_tabular_clean), 
    (df_images_stat, df_images_stat_clean), 
    (df_images_semantic, df_images_semantic_clean); format="tex", suffix="_grouptype")

ff(df) = filter(x -> x.modelname in representatives, df)

# one high row with all models
representatives=["ocsvm", "vae_full", "vae", "wae_full" , "wae", "fmgan", "vae_ocsvm", "knn", "aae_full", "aae", "RealNVP", "fAnoGAN", "DeepSVDD"]
df_tab = df_tabular |> ff |> copy, df_tabular_clean |> ff |> copy;
df_img_stat = df_images_stat |> ff |> copy, df_images_stat_clean |> ff |> copy;
df_img_sem = df_images_semantic |> ff |> copy, df_images_semantic_clean |> ff |> copy;

plot_knowledge_combined(df_tab, df_img_stat, df_img_sem; 
                    grouptype=false, format="tex", suffix="_repre_new_per_ac_one_all")
@info "plot_combine_knowledge_type"

@info "----------------- DONE ---------------------"
