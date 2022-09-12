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

sgad_models = ["DeepSVDD", "fAnoGAN", "fmgan", "fmganpy", "vae", "cgn", "sgvae", "vaegan", "sgvaegan"]
sgad_models_alpha = ["DeepSVDD", "fAnoGAN", "fmgan", "fmganpy", "vae", "cgn", "vaegan", "sgvae", "sgvaegan", 
    "sgvae_alpha", "sgvaegan_alpha"]
sgad_models_alias = [MODEL_ALIAS[n] for n in sgad_models_alpha]
DOWNSAMPLE = 50

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
            sorted_models = vcat(["dataset"], [x for x in sgad_models_alias if x in names(rt)])
            rt = rt[!,sorted_models]

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
        @info "Saved $f \n"
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

# alpha
df_images_alpha = load(datadir("sgad_alpha_evaluation_kp/images_leave-one-in_eval.bson"))[:df];
#df_images_alpha = load(datadir("sgad_alpha_evaluation_kp/images_leave-one-in_eval_converted.bson"))[:df];
apply_aliases!(df_images_alpha, col="dataset", d=DATASET_ALIAS) # rename
#filter!(r->r.modelname == "sgvae_robreg", df_images_alpha)
#df_images_alpha.modelname .= "sgvae_alpha"

# here select the best model and glue it to the normal df
prow = copy(df_images[1:1,:])
for dataset in unique(df_images_alpha.dataset)
    for ac in unique(df_images_alpha.anomaly_class)
        for model in ["sgvae_", "sgvaegan_"]
            subdf = filter(r->r.dataset==dataset && 
                r.anomaly_class==ac && 
                !isnan(r.tst_auc) &&
                occursin(model, r.modelname), df_images_alpha) 
            if size(subdf, 1) == 0
                continue
            end
            imax = argmax(subdf.val_auc)
            r = subdf[imax,:]
            prow.modelname = model*"alpha"
            prow.dataset = dataset
            prow.anomaly_class = ac
            prow.tst_auc = r.tst_auc
            prow.val_auc = r.val_auc
            prow.seed = 1
            df_images = vcat(df_images, prow)
        end
    end
end

# this generates the overall tables (aggregated by datasets)
df_images_target, _ = _split_image_datasets(df_images, TARGET_DATASETS);
df_images_target_nonnan = filter(r-> !isnan(r.val_auc), df_images_target)
prefix="images_loi"
suffix=""
modelnames = unique(df_images_target_nonnan.modelname)
downsample = Dict(zip(modelnames, repeat([DOWNSAMPLE], length(modelnames))))
rts = basic_summary_table(df_images_target_nonnan, outdir, prefix=prefix, suffix=suffix,
    downsample=downsample)
save_selection("$(datadir())/evaluation/$(outdir)/$(prefix)_auc_auc_maxmean$(suffix).csv", 
    rts[1], sgad_models_alias)

##### LOI images per AC
# this should generate the above tables split by anomaly classes
for d in Set(TARGET_DATASETS)
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
        sorted_models = vcat(["dataset"], [x for x in sgad_models_alias if x in names(rt)])
        rt = rt[!,sorted_models]

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
modelnames = unique(df_images_target_nonnan.modelname)
downsample = Dict(zip(modelnames, repeat([DOWNSAMPLE], length(modelnames))))
rts = basic_summary_table_per_ac(df_images_target_nonnan, outdir, prefix=prefix, suffix=suffix,
    downsample=downsample)
save_selection("$(datadir())/evaluation/$(outdir)/$(prefix)_auc_auc_autoagg$(suffix).csv", 
    rts[1], sgad_models_alias)

##### MVTEC
# now let's do the same for mvtec results
df_mvtec = load(datadir("evaluation/images_mvtec_eval_all.bson"))[:df];
apply_aliases!(df_mvtec, col="dataset", d=DATASET_ALIAS)
df_mvtec = filter(r->r.modelname in sgad_models, df_mvtec)
df_mvtec = filter(r->!(r.dataset in ["grid", "wood"]), df_mvtec)
df_mvtec_nonnan = filter(r-> !isnan(r.val_auc), df_mvtec)

#
df_mvtec_alpha = load(datadir("sgad_alpha_evaluation_kp/images_mvtec_eval.bson"))[:df];
apply_aliases!(df_mvtec_alpha, col="dataset", d=DATASET_ALIAS) # rename
filter!(r->r.modelname == "sgvae_robreg", df_mvtec_alpha)
df_mvtec_alpha.modelname .= "sgvae_alpha"
df_mvtec_alpha.dataset[df_mvtec_alpha.dataset .== "metal_nut"] .= "nut"

# here select the best model and glue it to the normal df
prow = copy(df_mvtec[1:1,:])
for dataset in unique(df_mvtec_alpha.dataset)
    for seed in unique(df_mvtec_alpha.seed)
        subdf = filter(r->r.dataset==dataset && r.seed==seed && !isnan(r.tst_auc), df_mvtec_alpha) 
        imax = argmax(subdf.val_auc)
        r = subdf[imax,:]
        prow.modelname = "sgvae_alpha"
        prow.dataset = dataset
        prow.seed = r.seed
        prow.tst_auc = r.tst_auc
        prow.val_auc = r.val_auc
        prow.seed = 1
        df_mvtec_nonnan = vcat(df_mvtec_nonnan, prow)
    end
end

prefix="images_mvtec"
suffix=""
modelnames = unique(df_mvtec_nonnan.modelname)
downsample = Dict(zip(modelnames, repeat([DOWNSAMPLE], length(modelnames))))
rts = basic_summary_table(df_mvtec_nonnan, outdir, prefix=prefix, suffix=suffix,
    downsample=downsample)
save_selection("$(datadir())/evaluation/$(outdir)/$(prefix)_auc_auc_maxmean$(suffix).csv", 
    rts[1], sgad_models_alias)

end
