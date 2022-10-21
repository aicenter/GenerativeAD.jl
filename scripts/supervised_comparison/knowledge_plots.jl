using DrWatson
using FileIO, BSON, DataFrames
using PrettyTables
using Statistics
using StatsBase
import PGFPlots
using CSV
using Suppressor
using Random
using ArgParse

using GenerativeAD.Evaluation: MODEL_ALIAS, DATASET_ALIAS, MODEL_TYPE, apply_aliases!
using GenerativeAD.Evaluation: _prefix_symbol, aggregate_stats_mean_max, aggregate_stats_max_mean
using GenerativeAD.Evaluation: PAT_METRICS, PATN_METRICS, PAC_METRICS, BASE_METRICS, TRAIN_EVAL_TIMES
using GenerativeAD.Evaluation: rank_table, print_rank_table, latex_booktabs, convert_anomaly_class

s = ArgParseSettings()
@add_arg_table! s begin
   "classes_val"
        default = 4
        arg_type = Int
        help = "no. validation anomalous classes"
end
parsed_args = parse_args(ARGS, s)
@unpack classes_val = parsed_args
classes_str = (classes_val == 4) ? "" : "_$(classes_val)v$(9-classes_val)"
AUC_METRICS = ["auc_100", "auc_50", "auc_20", "auc_10", "auc_5", "auc_2", "auc_1", "auc_05", "auc_02", "auc_01"]
AUC_METRICS_NAMES = ["\$AUC@\\%100\$", "\$AUC@\\%50\$", "\$AUC@\\%20\$", "\$AUC@\\%10\$", "\$AUC@\\%5\$", 
	"\$AUC@\\%2\$", "\$AUC@\\%1\$", "\$AUC@\\%0.5\$", "\$AUC@\\%0.2\$", "\$AUC@\\%0.1\$"]
AUCP_METRICS = map(x-> "auc_100_$(x)", [100, 50, 20, 10, 5, 2, 1])
AUCP_METRICS_NAMES = ["\$AUC@\\%100\$", "\$AUC@\\%50\$", "\$AUC@\\%20\$", "\$AUC@\\%10\$", "\$AUC@\\%5\$", 
	"\$AUC@\\%2\$", "\$AUC@\\%1\$"]
DOWNSAMPLE = 50
dseed = 40
topn = 1

include("../evaluation/utils/ranks.jl")
include("../evaluation/utils/utils.jl")
outdir = "result_tables"

models = ["classifier", "DeepSVDD", "fAnoGAN", "fmganpy10", "vae", "cgn", "cgn_0.2", "vaegan10", "sgvae",
 "sgvae_alpha", "sgvaegan10", "sgvaegan10_alpha", "sgvaegan100", "sgvaegan100_alpha"]
models_alias = ["classifier", "dsvd", "fano", "fmgn", "vae", "cgn", "cgn2", "vgn10", "sgvae", "sgvaea", 
    "sgvgn10", "sgvgn10a", "sgvgn100", "sgvgn100a"]
round_results = false

# LOI basic tables
df_images = load(datadir("supervised_comparison$(classes_str)/images_leave-one-in_eval.bson"))[:df];
df_images = setup_classic_models(df_images)
for model in ["sgvae_", "sgvaegan10_", "sgvaegan100_"]
    df_images.modelname[map(r->occursin(model, r.modelname), eachrow(df_images))] .= model*"alpha"
end
df_images = filter(r->r.modelname in models, df_images)
# rename the models
for (m,a) in zip(models, models_alias)
    df_images.modelname[df_images.modelname .== m] .= a
end

# metric definition
val_metrics = [:val_auc_01_100, :val_auc_02_100, :val_auc_05_100, :val_auc_1_100, :val_auc_2_100,
    :val_auc_5_100, :val_auc_10_100, :val_auc_20_100, :val_auc_50_100, :val_auc_100_100, :val_auc]
tst_metrics = [:tst_auc_01_100, :tst_auc_02_100, :tst_auc_05_100, :tst_auc_1_100, :tst_auc_2_100,
    :tst_auc_5_100, :tst_auc_10_100, :tst_auc_20_100, :tst_auc_50_100, :tst_auc_100_100, :tst_auc]

function collect_plot_points(modelname, dataset, ac, seed, df, val_metrics, tst_metrics)
    # filter the model, dataset and anomaly class
    subdf = filter(r->
        r.modelname == modelname &&
        r.dataset == dataset && 
        r.seed == seed &&
        r.anomaly_class == ac,
        df
        )

    res = []
    for (val_metric, tst_metric) in zip(val_metrics, tst_metrics)
        _subdf = filter(r->
            !isnan(r[val_metric]) &&
            !isnan(r[tst_metric]),
            subdf
            )
        n = size(_subdf,1)
        if n == 0
            push!(res, NaN)
        else
            # subsample the models
            Random.seed!(dseed)
            inds = sample(1:n, min(n, DOWNSAMPLE), replace=false)
            _subdf = _subdf[inds, :]
            Random.seed!()
            sortinds = sortperm(_subdf[val_metric], rev=true)
            imax = sortinds[min(topn, n)]
            push!(res, _subdf[tst_metric][imax])
        end
    end
    return res
end

res_df = DataFrame(
    :modelname => String[],
    :dataset => String[],
    :anomaly_class => Int[],
    :seed => Int[]
    )
for m in tst_metrics
    res_df[:,m] = Float32[]
end

for modelname in models_alias
    for dataset in ["cifar10", "svhn2", "wmnist", "coco"]
        for ac in 1:10
            for seed in 1:1
                res = collect_plot_points(modelname, dataset, ac, seed, df_images, val_metrics, tst_metrics)
                push!(res_df, vcat([modelname, dataset, ac, seed], res))
            end
        end
    end
end

save_str = (classes_val == 4) ? "_4v5" : classes_str
f = datadir("evaluation/result_tables/supervised_comparison$(save_str).csv")
CSV.write(f, res_df)
@info "Written result to $f"
