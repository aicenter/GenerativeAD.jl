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
outdir = "result_tables"
df_images = load(datadir("evaluation/images_leave-one-in_eval_all.bson"))[:df];
apply_aliases!(df_images, col="dataset", d=DATASET_ALIAS)

TARGET_DATASETS = Set(["cifar10", "svhn2", "wmnist"])

# splits single and multi class image datasets into "statistic" and "semantic" anomalies
_split_image_datasets(df, dt) = (
            filter(x -> x.dataset in dt, df), 
            filter(x -> ~(x.dataset in dt), df)
        )

function basic_summary_table(df, dir; suffix="", prefix="", downsample=Dict{String, Int}())
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

df_images_target, _ = _split_image_datasets(df_images, TARGET_DATASETS);
# this generates the overall tables (aggregated by datasets)
rts = basic_summary_table(df_images_target, outdir, prefix="images_loi", suffix="")

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

rts = basic_summary_table_per_ac(df_images_target, outdir, prefix="images_loi", suffix="_per_ac")

# now let's do the same for mvtec results
df_mvtec = load(datadir("evaluation/images_mvtec_eval_all.bson"))[:df];
apply_aliases!(df_mvtec, col="dataset", d=DATASET_ALIAS)
rts = basic_summary_table(df_mvtec, outdir, prefix="images_mvtec", suffix="")
