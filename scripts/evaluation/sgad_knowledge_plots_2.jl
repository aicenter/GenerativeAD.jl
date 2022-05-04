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

AUC_METRICS = ["auc_100", "auc_50", "auc_20", "auc_10", "auc_5", "auc_2", "auc_1", "auc_05", "auc_02", "auc_01"]
AUC_METRICS_NAMES = ["\$AUC@\\%100\$", "\$AUC@\\%50\$", "\$AUC@\\%20\$", "\$AUC@\\%10\$", "\$AUC@\\%5\$", 
	"\$AUC@\\%2\$", "\$AUC@\\%1\$", "\$AUC@\\%0.5\$", "\$AUC@\\%0.2\$", "\$AUC@\\%0.1\$"]
AUCP_METRICS = map(x-> "auc_100_$(x)", [100, 50, 20, 10, 5, 2, 1])
AUCP_METRICS_NAMES = ["\$AUC@\\%100\$", "\$AUC@\\%50\$", "\$AUC@\\%20\$", "\$AUC@\\%10\$", "\$AUC@\\%5\$", 
	"\$AUC@\\%2\$", "\$AUC@\\%1\$"]


include("./utils/ranks.jl")
outdir = "result_tables"

sgad_models = ["DeepSVDD", "fAnoGAN", "fmgan", "vae", "cgn", "sgvae", "sgvae_alpha"
#, "sgvae_alpha_knn", "sgvae_alpha_normal", 
# "sgvae_alpha_normal_logpx", "sgvae_alpha_kld", "sgvae_alpha_auc"
]
sgad_models_alias = [MODEL_ALIAS[n] for n in sgad_models]
sgad_alpha_models = ["sgvae_alpha"]
#, "sgvae_alpha_knn", "sgvae_alpha_normal", "sgvae_alpha_normal_logpx",  "sgvae_alpha_kld", "sgvae_alpha_auc"]

TARGET_DATASETS = Set(["cifar10", "svhn2", "wmnist"])

# LOI basic tables
source_images = datadir("evaluation_kp/images_leave-one-in_cache")
fs_images = readdir(source_images, join=true)
df_images = reduce(vcat, map(x->load(x)[:df], fs_images))

df_images = load(datadir("evaluation_kp/images_leave-one-in_eval.bson"))[:df];
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
df_mvtec = load(datadir("evaluation_kp/images_mvtec_eval.bson"))[:df];
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

"""
# also, add the clean sgvae alpha lines - these are the same as the ones from sgvae
function add_alpha_clean(df)
    subdf = filter(r->r.modelname == "sgvae", df)
    for suffix in ["_alpha", "_alpha_knn", "_alpha_kld", "_alpha_normal", "_alpha_normal_logpx", "_alpha_auc"]
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
"""


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

## setup
mn = "AUC"
metric = :auc
val_metric = _prefix_symbol("val", metric)
tst_metric = _prefix_symbol("tst", metric)
titles = ["semantic", "wmnist", "mvtec"]

# aggragated cols
non_agg_cols = ["modelname","dataset","anomaly_class","phash","parameters","seed","npars",
	"fs_fit_t","fs_eval_t"]
agg_cols = filter(x->!(x in non_agg_cols), names(df_images))
auto_f(df,crit) = aggregate_stats_auto(df, crit; agg_cols=agg_cols)
maxmean_f(df,crit) = aggregate_stats_max_mean(df, crit; agg_cols=[])

# first create the knowledge plot data for the changing level of anomalies
# at different percentages of normal data
cnames = reverse(AUC_METRICS_NAMES)

@suppress_err begin
for level in [100, 50, 10]
	criterions = reverse(_prefix_symbol.("val", map(x->x*"_$level",  AUC_METRICS)))
	extended_criterions = vcat(criterions, [val_metric])
	extended_cnames = vcat(["clean"], vcat(cnames, ["\$$(mn)_{val}\$"]))

	ranks_dfs = map(enumerate(zip(titles,
	        [(df_semantic, df_semantic_clean), 
	            (df_wmnist, df_wmnist_clean), 
	            (df_mvtec, df_mvtec_clean)]))) do (i, (title, (df, df_clean)))

	    ranks_inc, metric_means_inc = _incremental_rank(df, extended_criterions, auto_f)    
	    if size(df_clean,1) > 0
	        ranks_clean, metric_means_clean = _incremental_rank(df_clean, [val_metric], maxmean_f)
	        if !("cgn" in names(metric_means_clean))
	            metric_means_clean[:cgn] = NaN
	        end
	        ranks_all, metric_means_all = vcat(ranks_clean, ranks_inc; cols=:intersect), 
	        vcat(metric_means_clean, metric_means_inc; cols=:intersect)
	    else
	        ranks_all, metric_means_all = ranks_inc, metric_means_inc
	    end

	    # reorder table on tabular data as there is additional class of models (flows)
	    # one can do this manually at the end
	    models = names(ranks_all)
	    f = joinpath(datadir(), "evaluation", outdir, "knowledge_plot_v2_$(title)_ano_$(level).csv")
	    println("saving to $f")
	    CSV.write(f, metric_means_all)
	    ranks_all, metric_means_all
	end
end
end

# then do it again but for 50/50 anomaly/normal ratios and changing amount of data
cnames = reverse(AUCP_METRICS_NAMES)
criterions = reverse(_prefix_symbol.("val", AUCP_METRICS))
extended_criterions = vcat(criterions, [val_metric])
extended_cnames = vcat(["clean"], vcat(cnames, ["\$$(mn)_{val}\$"]))

@suppress_err begin

ranks_dfs = map(enumerate(zip(titles,
        [(df_semantic, df_semantic_clean), 
            (df_wmnist, df_wmnist_clean), 
            (df_mvtec, df_mvtec_clean)]))) do (i, (title, (df, df_clean)))

    ranks_inc, metric_means_inc = _incremental_rank(df, extended_criterions, auto_f)    
    if size(df_clean,1) > 0
        ranks_clean, metric_means_clean = _incremental_rank(df_clean, [val_metric], maxmean_f)
        if !("cgn" in names(metric_means_clean))
            metric_means_clean[:cgn] = NaN
        end
        ranks_all, metric_means_all = vcat(ranks_clean, ranks_inc; cols=:intersect), 
        vcat(metric_means_clean, metric_means_inc; cols=:intersect)
    else
        ranks_all, metric_means_all = ranks_inc, metric_means_inc
    end

    # reorder table on tabular data as there is additional class of models (flows)
    # one can do this manually at the end
    models = names(ranks_all)
    f = joinpath(datadir(), "evaluation", outdir, "knowledge_plot_v2_$(title)_prop.csv")
    println("saving to $f")
    CSV.write(f, metric_means_all)
    ranks_all, metric_means_all
end

end


level = 100
criterions = reverse(_prefix_symbol.("val", map(x->x*"_$level",  AUC_METRICS)))
extended_criterions = vcat(criterions, [val_metric])
extended_cnames = vcat(["clean"], vcat(cnames, ["\$$(mn)_{val}\$"]))
criterion = criterions[end]

ranks_inc, metric_means_inc = _incremental_rank(df_semantic, extended_criterions, auto_f)    

df = copy(df_semantic)
agg = auto_f
df_nonnan = filter(r->!(isnan(r[criterion])), df)

df_agg = agg(df_nonnan, criterion)
df_agg[:,[criterion, :tst_auc, :dataset, :modelname]]
