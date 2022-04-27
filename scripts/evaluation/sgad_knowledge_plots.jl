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

@suppress_err begin

include("./utils/ranks.jl")
outdir = "result_tables"

sgad_models = ["DeepSVDD", "fAnoGAN", "fmgan", "vae", "cgn", "sgvae", "sgvae_alpha",
    "sgvae_alpha_knn", "sgvae_alpha_normal", "sgvae_alpha_normal_logpx", "sgvae_alpha_kld",
    "sgvae_alpha_auc"]
sgad_models_alias = [MODEL_ALIAS[n] for n in sgad_models]
sgad_alpha_models = ["sgvae_alpha", "sgvae_alpha_knn", "sgvae_alpha_normal", "sgvae_alpha_normal_logpx", 
    "sgvae_alpha_kld", "sgvae_alpha_auc"]

TARGET_DATASETS = Set(["cifar10", "svhn2", "wmnist"])

# LOI basic tables
df_images = load(datadir("evaluation/images_leave-one-in_eval_all.bson"))[:df];
apply_aliases!(df_images, col="dataset", d=DATASET_ALIAS) # rename
# filter out only the interesting models
df_images = filter(r->r.modelname in sgad_models, df_images)

# this generates the overall tables (aggregated by datasets)
df_images_target, _ = _split_image_datasets(df_images, TARGET_DATASETS);

##### KNOWLEDGE PLOTS
# now let's load the clean dataframes and put together some knowledge plots
orig_path = "/home/skvarvit/generativead/GenerativeAD.jl/data"
df_images_loi_clean = load(datadir("evaluation/images_leave-one-in_clean_val_final_eval_all.bson"))[:df];
df_images_loi_clean = filter(r->r.modelname in sgad_models, df_images_loi_clean)

df_images_loi_clean[df_images_loi_clean.fit_t .=== nothing, :fit_t] .= 1.0; # ConvSkipGANomaly is missing the fit_t
apply_aliases!(df_images_loi_clean, col="dataset", d=DATASET_ALIAS)
for d in Set(["cifar10", "svhn2", "wmnist"])
    mask = (df_images_loi_clean.dataset .== d)
    df_images_loi_clean[mask, :dataset] .= df_images_loi_clean[mask, :dataset] .* ":" .* convert_anomaly_class.(df_images_loi_clean[mask, :anomaly_class], d)
    df_images_loi_clean[mask, :anomaly_class] .= 1 # it has to be > 0, because otherwise we get too many warnings from the aggregate_stats_max_mean
end

# mvtec
df_mvtec = load(datadir("evaluation/images_mvtec_eval_all.bson"))[:df];
apply_aliases!(df_mvtec, col="dataset", d=DATASET_ALIAS)
df_mvtec = filter(r->r.modelname in sgad_models, df_mvtec)
df_mvtec = filter(r->!(r.dataset in ["grid", "wood"]), df_mvtec)
df_images_mvtec_clean = load(datadir("evaluation/images_mvtec_clean_val_final_eval_all.bson"))[:df];
df_images_mvtec_clean = filter(r->r.modelname in sgad_models, df_images_mvtec_clean)
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

# also, add the clean sgvae alpha lines - these are the same as the ones from sgvae
function add_alpha_clean(df)
    subdf = filter(r->r.modelname == "sgvae", df)
    for suffix in ["_alpha", "_alpha_knn", "_alpha_kld", "_alpha_normal", "_alpha_normal_logpx"]
        subdf.modelname .= "sgvae" * suffix
        df = vcat(df, copy(subdf))
    end
    df
end
df_semantic_clean = add_alpha_clean(df_semantic_clean)
df_wmnist_clean = add_alpha_clean(df_wmnist_clean)
df_mvtec_clean = add_alpha_clean(df_mvtec_clean)

# remove duplicate rows from mvtec clean
models = df_mvtec_clean.modelname
seeds = df_mvtec_clean.seed
datasets = df_mvtec_clean.dataset
subdfs = []
for model in unique(models)
    for dataset in unique(datasets)
        for seed in unique(seeds)
            inds = (models .== model) .& (seeds .== seed) .& (datasets .== dataset)
            if sum(inds) > 0
                subdf = df_mvtec_clean[inds,:]
                #subdf = subdf[argmax(subdf.tst_auc), :]
                subdf = subdf[1, :]
                push!(subdfs, DataFrame(subdf))
            end
        end
    end
end
df_mvtec_clean = vcat(subdfs...)
# filter out grid and wood
df_mvtec_clean = filter(r->!(r.dataset in ["wood", "grid"]), df_mvtec_clean)

function _incremental_rank(df, criterions, agg)
    ranks, metric_means = [], []
    for criterion in criterions
        df_nonnan = filter(r->!(isnan(r[criterion])), df)
        if size(df_nonnan, 1) > 0
            df_agg = agg(df_nonnan, criterion)

            # some model might be missing
            nr = size(df_agg,2)
            modelnames = df_agg.modelname
            if !("sgvae_alpha" in modelnames)
                for dataset in unique(df_agg.dataset)
                    for m in sgad_alpha_models
                        df_agg = push!(df_agg, vcat(repeat([NaN], nr-2), [dataset, m]))
                    end
                end
            end

            apply_aliases!(df_agg, col="modelname", d=MODEL_RENAME)
            apply_aliases!(df_agg, col="modelname", d=MODEL_ALIAS)
            sort!(df_agg, [:dataset, :modelname])
            rt = rank_table(df_agg, tst_metric)
            mm = DataFrame([Symbol(model) => mean(rt[1:end-3, model]) for model in names(rt)[2:end]])
            push!(ranks, rt[end:end, 2:end])
            push!(metric_means, mm)
        end
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

## setup
mn = "AUC"
metric = :auc
val_metric = _prefix_symbol("val", metric)
tst_metric = _prefix_symbol("tst", metric)

cnames = PAT_METRICS_NAMES
criterions = _prefix_symbol.("val", PAT_METRICS)

extended_criterions = vcat(criterions, [val_metric])
extended_cnames = vcat(["clean"], vcat(cnames, ["\$$(mn)_{val}\$"]))
titles = ["semantic", "wmnist", "mvtec"]

#ab_plots = map(enumerate([(df_semantic, df_semantic_clean), (df_wmnist, df_wmnist_clean), (df_mvtec, df_mvtec_clean)])) do (i, (df, df_clean))
ranks_dfs = map(enumerate(zip(titles,
        [(df_semantic, df_semantic_clean), 
            (df_wmnist, df_wmnist_clean), 
            (df_mvtec, df_mvtec_clean)]))) do (i, (title, (df, df_clean)))
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