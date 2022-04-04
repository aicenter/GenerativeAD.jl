using DrWatson
using FileIO, BSON, DataFrames
using PrettyTables
using Statistics
using StatsBase
import PGFPlots
using CSV
using Suppressor

using GenerativeAD.Evaluation: MODEL_ALIAS, DATASET_ALIAS, MODEL_TYPE, apply_aliases!
using GenerativeAD.Evaluation: _prefix_symbol, aggregate_stats_mean_max, aggregate_stats_max_mean
using GenerativeAD.Evaluation: PAT_METRICS, PATN_METRICS, PAC_METRICS, BASE_METRICS, TRAIN_EVAL_TIMES
using GenerativeAD.Evaluation: rank_table, print_rank_table, latex_booktabs, convert_anomaly_class

const PAT_METRICS_NAMES = ["\$PR@\\%0.01\$","\$PR@\\%0.1\$","\$PR@\\%1\$","\$PR@\\%5\$","\$PR@\\%10\$","\$PR@\\%20\$"]

# this is to suppress warnings
@suppress_err begin

include("./utils/ranks.jl")
outdir = "result_tables"
df_images = load(datadir("evaluation/images_leave-one-in_eval_all.bson"))[:df];
apply_aliases!(df_images, col="dataset", d=DATASET_ALIAS)

plot_models = ["dsvd", "fano", "fmgn", "vae", "cgn", "sgvae", "sgvae_alpha"]

TARGET_DATASETS = Set(["cifar10", "svhn2", "wmnist"])

# splits single and multi class image datasets into "statistic" and "semantic" anomalies
_split_image_datasets(df, dt) = (
            filter(x -> x.dataset in dt, df), 
            filter(x -> ~(x.dataset in dt), df)
        )

function basic_summary_table(df, dir; suffix="", prefix="", downsample=Dict{String, Int}(), )
    agg_names = ["maxmean"]
    agg_funct = [aggregate_stats_max_mean]
    rts = []
    for (name, agg) in zip(agg_names, agg_funct)
        for metric in [:auc]
            val_metric = _prefix_symbol("val", metric)
            tst_metric = _prefix_symbol("tst", metric)    
            
            _, rt = sorted_rank(df, agg, val_metric, tst_metric, downsample)

            rt[end-2, 1] = "\$\\sigma_1\$"
            rt[end-1, 1] = "\$\\sigma_{10}\$"
            rt[end, 1] = "rnk"

            file = "$(datadir())/evaluation/$(dir)/$(prefix)_$(metric)_$(metric)_$(name)$(suffix).txt"
            open(file, "w") do io
                print_rank_table(io, rt; backend=:txt) # or :tex
            end
            @info "saved to $file"
            push!(rts, rt)
        end
    end
    rts
end

function save_selection(f, rt, plot_models)
    try
        model_cols = vcat([:dataset], Symbol.(plot_models))
        CSV.write(f, rt[model_cols])
    catch e
        if typeof(e) == ArgumentError
            @info "One of the models is not present in the DataFrame to be saved into $f"
        else
            rethrow(e)
        end
    end
    f
end

##### LOI images 
# this generates the overall tables (aggregated by datasets)
df_images_target, _ = _split_image_datasets(df_images, TARGET_DATASETS);
prefix="images_loi"
suffix=""
rts = basic_summary_table(df_images_target, outdir, prefix=prefix, suffix=suffix)
save_selection("$(datadir())/evaluation/$(outdir)/$(prefix)_auc_auc_maxmean$(suffix).csv", 
    rts[1], plot_models)

# this is to be used for further distinguishing performance based on some hyperparam values
perf_plot_models = ["dsvd", "fano", "fmgn", "vae", "cgn", "sgvae", "sgvae_d", "sgvae_i"]

suffix = "_sgvae_latent_structure"
subdf = filter(r->r.modelname=="sgvae", df_images_target)
params = map(x->get(parse_savename(x)[2], "latent_structure", ""), subdf.parameters)
subdf.modelname[params .== "mask"] .= "sgvae_d"
subdf.modelname[params .!= "mask"] .= "sgvae_i"
perf_df_images_target = vcat(subdf, df_images_target)
rts = basic_summary_table(perf_df_images_target, outdir, prefix=prefix, suffix=suffix)
save_selection("$(datadir())/evaluation/$(outdir)/$(prefix)_auc_auc_maxmean$(suffix).csv", 
    rts[1], perf_plot_models)







##### LOI images per AC
# this should generate the above tables split by anomaly classes
for d in Set(["cifar10", "svhn2", "wmnist"])
    mask = (df_images_target.dataset .== d)
    df_images_target[mask, :dataset] .= df_images_target[mask, :dataset] .* ":" .* convert_anomaly_class.(df_images_target[mask, :anomaly_class], d)
    df_images_target[mask, :anomaly_class] .= 1 # it has to be > 0, because otherwise we get too many warnings from the aggregate_stats_max_mean
end

function basic_summary_table_per_ac(df, dir; suffix="", prefix="", downsample=Dict{String, Int}())
    rts = []   
    for metric in [:auc]
        val_metric = _prefix_symbol("val", metric)
        tst_metric = _prefix_symbol("tst", metric)    

        _, rt = sorted_rank(df, aggregate_stats_auto, val_metric, tst_metric, downsample)

        rt[end-2, 1] = "\$\\sigma_1\$"
        rt[end-1, 1] = "\$\\sigma_{10}\$"
        rt[end, 1] = "rnk"

        file = "$(datadir())/evaluation/$(dir)/$(prefix)_$(metric)_$(metric)_autoagg$(suffix).txt"
        open(file, "w") do io
            print_rank_table(io, rt; backend=:txt)
        end
        @info "saved to $file"
        push!(rts, rt)
    end
    rts
end

prefix="images_loi"
suffix="_per_ac"
rts = basic_summary_table_per_ac(df_images_target, outdir, prefix=prefix, suffix=suffix)
save_selection("$(datadir())/evaluation/$(outdir)/$(prefix)_auc_auc_autoagg$(suffix).csv", 
    rts[1], plot_models)


# this is to be used for further distinguishing performance based on some hyperparam values
for d in Set(["cifar10", "svhn2", "wmnist"])
    mask = (perf_df_images_target.dataset .== d)
    perf_df_images_target[mask, :dataset] .= perf_df_images_target[mask, :dataset] .* ":" .* convert_anomaly_class.(perf_df_images_target[mask, :anomaly_class], d)
    perf_df_images_target[mask, :anomaly_class] .= 1 # it has to be > 0, because otherwise we get too many warnings from the aggregate_stats_max_mean
end

prefix="images_loi"
suffix = "_sgvae_latent_structure_per_ac"
rts = basic_summary_table_per_ac(perf_df_images_target, outdir, prefix=prefix, suffix=suffix)
save_selection("$(datadir())/evaluation/$(outdir)/$(prefix)_auc_auc_autoagg$(suffix).csv", 
    rts[1], perf_plot_models)






##### MVTEC
# now let's do the same for mvtec results
df_mvtec = load(datadir("evaluation/images_mvtec_eval_all.bson"))[:df];
apply_aliases!(df_mvtec, col="dataset", d=DATASET_ALIAS)
prefix="images_mvtec"
suffix=""
rts = basic_summary_table(df_mvtec, outdir, prefix=prefix, suffix=suffix)
save_selection("$(datadir())/evaluation/$(outdir)/$(prefix)_auc_auc_maxmean$(suffix).csv", 
    rts[1], plot_models)

# do this for the different sgvae params
prefix="images_mvtec"
suffix = "_sgvae_latent_structure"

subdf = filter(r->r.modelname=="sgvae", df_mvtec)
params = map(x->get(parse_savename(x)[2], "latent_structure", ""), subdf.parameters)
subdf.modelname[params .== "mask"] .= "sgvae_d"
subdf.modelname[params .!= "mask"] .= "sgvae_i"
perf_df_mvtec = vcat(subdf, df_mvtec)
rts = basic_summary_table(perf_df_mvtec, outdir, prefix=prefix, suffix=suffix)
save_selection("$(datadir())/evaluation/$(outdir)/$(prefix)_auc_auc_maxmean$(suffix).csv", 
    rts[1], perf_plot_models)






##### KNOWLEDGE PLOTS
# now let's load the clean dataframes and put together some knowledge plots
orig_path = "/home/skvarvit/generativead/GenerativeAD.jl/data"
df_images_loi_clean = load(datadir("evaluation/images_leave-one-in_clean_val_final_eval_all.bson"))[:df];
#load(joinpath(orig_path, "evaluation/images_leave-one-in_clean_val_final_eval.bson"))[:df];
df_images_loi_clean[df_images_loi_clean.fit_t .=== nothing, :fit_t] .= 1.0; # ConvSkipGANomaly is missing the fit_t
apply_aliases!(df_images_loi_clean, col="dataset", d=DATASET_ALIAS)
for d in Set(["cifar10", "svhn2", "wmnist"])
    mask = (df_images_loi_clean.dataset .== d)
    df_images_loi_clean[mask, :dataset] .= df_images_loi_clean[mask, :dataset] .* ":" .* convert_anomaly_class.(df_images_loi_clean[mask, :anomaly_class], d)
    df_images_loi_clean[mask, :anomaly_class] .= 1 # it has to be > 0, because otherwise we get too many warnings from the aggregate_stats_max_mean
end

# mvtec
df_images_mvtec_clean = load(datadir("evaluation/images_mvtec_clean_val_final_eval_all.bson"))[:df];
#load(joinpath(orig_path, "evaluation/images_mvtec_clean_val_final_eval.bson"))[:df];
df_mvtec[:anomaly_class] = 1
df_images_mvtec_clean[:anomaly_class] = 1
apply_aliases!(df_images_mvtec_clean, col="dataset", d=DATASET_ALIAS)


# now differentiate them
df_semantic = filter(r->!(occursin("wmnist", r[:dataset])),df_images_target)
df_semantic_clean = filter(r->!(occursin("mnist", r[:dataset])),df_images_loi_clean)

df_wmnist = filter(r->(occursin("wmnist", r[:dataset])),df_images_target)
df_wmnist_clean = filter(r->(occursin("wmnist", r[:dataset])),df_images_loi_clean)

df_mvtec = df_mvtec
df_mvtec_clean = df_images_mvtec_clean


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

mn = "AUC"
metric = :auc
val_metric = _prefix_symbol("val", metric)
tst_metric = _prefix_symbol("tst", metric)
format = "pdf"

ctype = "pat"
cnames = PAT_METRICS_NAMES
criterions = _prefix_symbol.("val", PAT_METRICS)

extended_criterions = vcat(criterions, [val_metric])
extended_cnames = vcat(["clean"], vcat(cnames, ["\$$(mn)_{val}\$"]))
titles = ["semantic", "wmnist", "mvtec"]

ranks_clean, metric_means_clean = _incremental_rank(df_semantic_clean, [val_metric], aggregate_stats_max_mean)
ranks_inc, metric_means_inc = _incremental_rank(df_semantic, extended_criterions, aggregate_stats_auto)
ranks_all, metric_means_all = vcat(ranks_clean, ranks_inc; cols=:intersect), vcat(metric_means_clean, metric_means_inc; cols=:intersect)
models = names(ranks_all)

#ab_plots = map(enumerate([(df_semantic, df_semantic_clean), (df_wmnist, df_wmnist_clean), (df_mvtec, df_mvtec_clean)])) do (i, (df, df_clean))
ranks_dfs = map(enumerate(
    zip(titles,
        [(df_semantic, df_semantic_clean), (df_wmnist, df_wmnist_clean), (df_mvtec, df_mvtec_clean)]))) do (i, (title, (df, df_clean)))
    ranks_inc, metric_means_inc = _incremental_rank(df, extended_criterions, aggregate_stats_auto)
    
    if size(df_clean,1) > 0
        ranks_clean, metric_means_clean = _incremental_rank(df_clean, [val_metric], aggregate_stats_max_mean)
        if !("cgn" in names(metric_means_clean))
            metric_means_clean[:cgn] = NaN
        end
        ranks_all, metric_means_all = vcat(ranks_clean, ranks_inc; cols=:intersect), 
        vcat(metric_means_clean, metric_means_inc; cols=:intersect)
    else
        ranks_all, metric_means_all = ranks_inc, metric_means_inc
    end
    # @info("", ranks_clean, ranks_inc)
    # @info("", metric_means_clean, metric_means_inc)

    # reorder table on tabular data as there is additional class of models (flows)
    # one can do this manually at the end
    models = names(ranks_all)
    f = joinpath(datadir(), "evaluation", outdir, "knowledge_plot_$(title)_data.csv")
    @info "saving to $f"
    CSV.write(f, metric_means_all)
    ranks_all, metric_means_all
#    a, b = _plot(ranks_all, metric_means_all, extended_cnames, models, titles[i])
#    a, b
end

end