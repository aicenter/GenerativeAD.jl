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

sgad_models = ["DeepSVDD", "fAnoGAN", "fmgan", "vae", "cgn", "sgvae", "sgvae_alpha"]
sgad_models_alias = [MODEL_ALIAS[n] for n in sgad_models]
sgad_alpha_models = ["sgvae_alpha"]

TARGET_DATASETS = Set(["cifar10", "svhn2", "wmnist", "coco"])

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
            file = "$(datadir())/evaluation/$(dir)/$(prefix)_$(metric)_$(metric)_$(name)$(suffix).tex"
            open(file, "w") do io
                print_rank_table(io, rt; backend=:tex) # or :tex
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
df_images = load(datadir("evaluation/images_leave-one-in_eval_all.bson"))[:df];
apply_aliases!(df_images, col="dataset", d=DATASET_ALIAS) # rename
# filter out only the interesting models
df_images = filter(r->r.modelname in sgad_models, df_images)

# this generates the overall tables (aggregated by datasets)
df_images_target, _ = _split_image_datasets(df_images, TARGET_DATASETS);
df_images_target_nonnan = filter(r-> !isnan(r.val_auc), df_images_target)
prefix="images_loi"
suffix=""
rts = basic_summary_table(df_images_target_nonnan, outdir, prefix=prefix, suffix=suffix)
save_selection("$(datadir())/evaluation/$(outdir)/$(prefix)_auc_auc_maxmean$(suffix).csv", 
    rts[1], sgad_models_alias)

# this is to be used for further distinguishing performance based on some hyperparam values
perf_plot_models = ["dsvd", "fano", "fmgn", "vae", "cgn", "sgvae", "sgvae_d", "sgvae_i"]

suffix = "_sgvae_latent_structure"
subdf = filter(r->r.modelname=="sgvae", df_images_target_nonnan)
params = map(x->get(parse_savename(x)[2], "latent_structure", ""), subdf.parameters)
subdf.modelname[params .== "mask"] .= "sgvae_d"
subdf.modelname[params .!= "mask"] .= "sgvae_i"
perf_df_images_target = vcat(subdf, df_images_target_nonnan)
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
df_images_target_nonnan = filter(r-> !isnan(r.val_auc), df_images_target)

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
        file = "$(datadir())/evaluation/$(dir)/$(prefix)_$(metric)_$(metric)_autoagg$(suffix).tex"
        open(file, "w") do io
            print_rank_table(io, rt; backend=:tex)
        end
        @info "saved to $file"
        push!(rts, rt)
    end
    rts
end

prefix="images_loi"
suffix="_per_ac"
rts = basic_summary_table_per_ac(df_images_target_nonnan, outdir, prefix=prefix, suffix=suffix)
save_selection("$(datadir())/evaluation/$(outdir)/$(prefix)_auc_auc_autoagg$(suffix).csv", 
    rts[1], sgad_models_alias)


# this is to be used for further distinguishing performance based on some hyperparam values
"""
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
"""



##### MVTEC
# now let's do the same for mvtec results
df_mvtec = load(datadir("evaluation/images_mvtec_eval_all.bson"))[:df];
apply_aliases!(df_mvtec, col="dataset", d=DATASET_ALIAS)
df_mvtec = filter(r->r.modelname in sgad_models, df_mvtec)
df_mvtec = filter(r->!(r.dataset in ["grid", "wood"]), df_mvtec)
df_mvtec_nonnan = filter(r-> !isnan(r.val_auc), df_mvtec)

prefix="images_mvtec"
suffix=""
rts = basic_summary_table(df_mvtec_nonnan, outdir, prefix=prefix, suffix=suffix)
save_selection("$(datadir())/evaluation/$(outdir)/$(prefix)_auc_auc_maxmean$(suffix).csv", 
    rts[1], sgad_models_alias)

# do this for the different sgvae params
"""
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
"""

end
