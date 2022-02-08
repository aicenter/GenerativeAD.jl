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

include("./utils/ranks.jl")
outdir = "images_leave-one-in_eval"
df_images = load(datadir("evaluation/images_leave-one-in_eval_all.bson"))[:df];

SEMANTIC_IMAGE_ANOMALIES = Set(["CIFAR10", "SVHN2"])

# splits single and multi class image datasets into "statistic" and "semantic" anomalies
_split_image_datasets(df) = (
            filter(x -> x.dataset in SEMANTIC_IMAGE_ANOMALIES, df), 
            filter(x -> ~(x.dataset in SEMANTIC_IMAGE_ANOMALIES), df)
        )

function basic_summary_table(df, dir; suffix="", prefix="", downsample=Dict{String, Int}())
    agg_names = ["maxmean", "meanmax"]
    agg_funct = [aggregate_stats_max_mean, aggregate_stats_mean_max]
    rts = []
    for (name, agg) in zip(agg_names, agg_funct)
        for metric in [:auc, :tpr_5]
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
            push!(rts, rt)
        end
    end
    rts
end

df_images_semantic, df_images_stat = _split_image_datasets(df_images);
# this generates the overall tables (aggregated by datasets)
rts = basic_summary_table(df_images_semantic, outdir, prefix="images_semantic", suffix="")

# this should generate the above tables split by anomaly classes
df = df_images_semantic
apply_aliases!(df, col="dataset", d=DATASET_ALIAS)
for d in Set(["cifar10", "svhn2"])
    mask = (df.dataset .== d)
    df[mask, :dataset] .= df[mask, :dataset] .* ":" .* convert_anomaly_class.(df[mask, :anomaly_class], d)
    df[mask, :anomaly_class] .= 1 # it has to be > 0, because otherwise we get too many warnings from the aggregate_stats_max_mean
end

function basic_summary_table_per_ac(df, dir; suffix="", prefix="", downsample=Dict{String, Int}())
    rts = []   
    for metric in [:auc, :tpr_5]
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
        push!(rts, rt)
    end
    rts
end

rts = basic_summary_table_per_ac(df, outdir, prefix="images_semantic", suffix="_per_ac")
