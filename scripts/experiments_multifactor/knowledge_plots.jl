using DrWatson
using FileIO, BSON, DataFrames
using PrettyTables
using Statistics
using StatsBase
import PGFPlots
using CSV
using Suppressor
using Random

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
include("../evaluation/utils/utils.jl")

outdir = datadir("experiments_multifactor/evaluation_mf_normal/result_tables")
mkpath(outdir)
dseed = 42
DOWNSAMPLE = 50

# load the dfs
df_images = load(datadir("experiments_multifactor/evaluation_mf_normal/images_leave-one-in_eval.bson"))[:df]
df_sgvgna = load(datadir("experiments_multifactor/alpha_evaluation_mf_normal/sgvaegan100_eval.bson"))[:df]
df_sgvaea = load(datadir("experiments_multifactor/alpha_evaluation_mf_normal/sgvae_eval.bson"))[:df]
df_sgvgna.modelname .= "sgvaegan100_alpha"
df_sgvaea.modelname .= "sgvae_alpha"
df_alpha = vcat(df_sgvgna, df_sgvaea)

# alias
models = ["DeepSVDD", "fAnoGAN", "fmganpy10", "vae", "cgn", "vaegan10", "sgvae",
  "sgvaegan100", "sgvae_alpha", "sgvaegan100_alpha"]
models_alias = ["dsvd", "fano", "fmgn", "vae", "cgn", "vgn", "sgvae", "sgvgn", 
    "sgvaea", "sgvgna"]
round_results = false
TARGET_DATASETS = Set(["wmnist", "cocoplaces"])

# setup the dfs
df_images = setup_classic_models(df_images)
for (m,a) in zip(models, models_alias)
    df_images.modelname[df_images.modelname .== m] .= a
end
for (m,a) in zip(models, models_alias)
    df_alpha.modelname[df_alpha.modelname .== m] .= a
end
apply_aliases!(df_alpha, col="dataset", d=DATASET_ALIAS) # rename
prepare_alpha_df!(df_alpha)


# now create the column with anomaly factors
parameters = map(x->replace(x, "anomaly_factors" => "anomaly-factors"), df_images.parameters)
df_images[:anomaly_factors] = map(x->parse_savename(x)[2]["anomaly-factors"], parameters) 
df_images[:score_type] = ""
df_images[:latent_score_type] = ""

# compute number of afs for models
#for model in unique(df_images.modelname)
#    println(model)
#    for af in unique(df_images.anomaly_factors)
#        subdf = filter(r->r.modelname == model && r.anomaly_factors==af, df_images)
#        println("   $af : $(size(subdf,1))")
#    end
#    println("")
#end
df_alpha.anomaly_factors = Meta.parse.(string.(df_alpha.anomaly_factors))

# metrics
val_metrics = [:val_auc_01_100, :val_auc_02_100, :val_auc_05_100, :val_auc_1_100, :val_auc_2_100,
    :val_auc_5_100, :val_auc_10_100, :val_auc_20_100, :val_auc_50_100, :val_auc_100_100, :val_auc]
tst_metricsa = [:tst_auc_01_100, :tst_auc_02_100, :tst_auc_05_100, :tst_auc_1_100, :tst_auc_2_100,
    :tst_auc_5_100, :tst_auc_10_100, :tst_auc_20_100, :tst_auc_50_100, :tst_auc_100_100, :tst_auc]
tst_metrics = repeat([:tst_auc], length(val_metrics))

res_df = DataFrame(
    :modelname => String[],
    :dataset => String[],
    :anomaly_class => Int[],
    :seed => Int[],
    :anomaly_factors => Int[],
    )
for m in tst_metricsa
    res_df[:,m] = Float32[]
end

for modelname in models_alias
    for (dataset, afs) in zip(["wmnist", "coco"], ([1, 2, 3, 12, 13, 23, 123], [1, 2, 12]))
        for ac in 1:10
            for seed in 1:1
                for af in afs
                    tstms, subdf = if modelname in ["sgvaea", "sgvgna"]
                        subdf = filter(r->r.anomaly_factors == af, df_alpha)
                        tstms = tst_metricsa
                        tstms, subdf
                    else
                        subdf = filter(r->r.anomaly_factors == af, df_images)
                        tstms = tst_metrics
                        tstms, subdf
                    end
                    res = collect_plot_points(modelname, dataset, ac, seed, subdf, val_metrics, tstms)
                    push!(res_df, vcat([modelname, dataset, ac, seed, af], res))
                end
            end
        end
    end
end

outf = datadir("../notebooks_paper/julia_data/experiments_multifactor.csv")
CSV.write(outf, res_df)