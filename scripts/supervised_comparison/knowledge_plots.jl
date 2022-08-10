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


include("../evaluation/utils/ranks.jl")
outdir = "result_tables"

models = ["classifier", "sgvae_alpha"]
round_results = false

# LOI basic tables
df_images = load(datadir("supervised_comparison/images_leave-one-in_eval.bson"))[:df];
apply_aliases!(df_images, col="dataset", d=DATASET_ALIAS) # rename
df_images.modelname[df_images.modelname .== "sgvae_robreg"] .= "sgvaea"

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
        if size(_subdf,1) == 0
            push!(res, NaN)
        else
            imax = argmax(_subdf[val_metric])
            push!(res, subdf[tst_metric][imax])
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

for modelname in ["classifier", "sgvaea"]
    for dataset in ["cifar10", "svhn2", "wmnist", "coco"]
        for ac in 1:10
            for seed in 1:1
                res = collect_plot_points(modelname, dataset, ac, seed, df_images, val_metrics, tst_metrics)
                push!(res_df, vcat([modelname, dataset, ac, seed], res))
            end
        end
    end
end

f = datadir("evaluation/result_tables/supervised_comparison.csv")
CSV.write(f, res_df)
@info "Written result to $f"