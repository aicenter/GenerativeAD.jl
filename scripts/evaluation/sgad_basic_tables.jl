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
include("./utils/utils.jl")
outdir = "result_tables"

sgad_models = ["DeepSVDD", "fAnoGAN", "fmgan", "fmganpy", "fmganpy10", "vae", "cgn", "cgn_0.2", 
"cgn_0.3", "vaegan", "vaegan10", "sgvaegan", "sgvaegan_0.5", "sgvaegan10", "sgvaegan100", "sgvae", 
"sgvae_alpha", "sgvaegan_alpha"]
sgad_alpha_models = ["classifier", "sgvae_alpha", "sgvaegan_alpha"]
sgad_models_final = ["DeepSVDD", "fAnoGAN", "fmgan", "vae", "cgn", "vaegan10", "sgvae",  "sgvaegan100", 
    "sgvae_alpha", "sgvaegan_alpha"]
MODEL_ALIAS["cgn_0.2"] = "cgn2"
MODEL_ALIAS["cgn_0.3"] = "cgn3"
MODEL_ALIAS["sgvaegan_0.5"] = "sgvgn05"
MODEL_ALIAS["sgvaegan100"] = "sgvgn100"
MODEL_ALIAS["sgvaegan10_alpha"] = "sgvgn10a"
MODEL_ALIAS["sgvaegan100_alpha"] = "sgvgn100a"
TARGET_DATASETS = Set(["cifar10", "svhn2", "wmnist", "coco"])
round_results = false
DOWNSAMPLE = 50
val_metric = :val_auc_100_100
tst_metrica = :tst_auc_100_100 
tst_metric = :tst_auc

# LOI basic tables
df_images = load(datadir("evaluation_kp/images_leave-one-in_eval.bson"))[:df];
# filter out only the interesting models
df_images = filter(r->r.modelname in sgad_models, df_images)
# this generates the overall tables (aggregated by datasets)
df_images = setup_classic_models(df_images)

# LOI alpha scores
df_images_alpha = load(datadir("sgad_alpha_evaluation_kp/images_leave-one-in_eval.bson"))[:df];
#df_images_alpha = load(datadir("sgad_alpha_evaluation_kp/images_leave-one-in_eval_converted.bson"))[:df];
df_images_alpha = setup_alpha_models(df_images_alpha)

# now there is a little bit more differentiation here
df_images_alpha = differentiate_beta_1_10(df_images_alpha)
df_images = differentiate_early_stopping(df_images)
df_images_alpha = differentiate_sgvaegana(df_images_alpha)

function basic_summary_table(df, dir; suffix="", prefix="", downsample=Dict{String, Int}())
    agg_names = ["maxmean"]
    agg_funct = [aggregate_stats_max_mean]
    rts = []
    metric = :auc
    for (name, agg) in zip(agg_names, agg_funct)            
        _, rt = sorted_rank(df, agg, val_metric, tst_metric, downsample, 
            agg_cols=[string(val_metric), string(tst_metric)])
        sorted_models = vcat(["dataset"], [x for x in models_alias if x in names(rt)])
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
# on coco we use sgvaegan10alpha, otherwise we use sgvaegan100alpha
filter!(r->!(
    r.modelname == "sgvaegan100_alpha" && 
    r.dataset in ["coco"]), df_images_alpha)
filter!(r->!(
    r.modelname == "sgvaegan10_alpha" && 
    r.dataset in ["wmnist", "svhn2", "cifar10"]), df_images_alpha)
filter!(r->r.modelname in ["sgvae_alpha", "sgvaegan10_alpha", "sgvaegan100_alpha"], df_images_alpha)
# now rename it all to sgvaegan_alpha
df_images_alpha.modelname[map(x->occursin("sgvaegan", x), df_images_alpha.modelname)] .= "sgvaegan_alpha"

# on svhn we use cgn2, otherwise cgn
filter!(r->!(
    r.modelname in ["cgn", "cgn_0.3"] && 
    r.dataset in ["svhn2"]), df_images)
# now rename it all to sgvaegan_alpha
df_images.modelname[map(x->occursin("cgn", x), df_images.modelname)] .= "cgn"

# on cifar, we use fmgan, otherwise we use fmganpy10
filter!(r->!(
    r.modelname in ["fmganpy", "fmganpy10"] && 
    r.dataset in ["cifar10"]), df_images)
filter!(r->!(
    r.modelname in ["fmgan", "fmganpy"] && 
    r.dataset in ["wmnist", "svhn2", "coco"]), df_images)
df_images.modelname[map(x->occursin("fmgan", x), df_images.modelname)] .= "fmgan"

# here select the best model and glue it to the normal df
prow = copy(df_images[1:1,:])
for dataset in unique(df_images_alpha.dataset)
    for ac in unique(df_images_alpha.anomaly_class)
        for model in ["sgvae_", "sgvaegan_"]
            subdf = filter(r->
                r.dataset==dataset && 
                r.anomaly_class==ac && 
                !isnan(r[tst_metrica]) &&
                !isnan(r[val_metric]) &&
                occursin(model, r.modelname), df_images_alpha) 
            if size(subdf, 1) == 0
                continue
            end
            imax = argmax(subdf[:,val_metric])
            r = subdf[imax,:]
            prow.modelname = model*"alpha"
            prow.dataset = dataset
            prow.anomaly_class = ac
            prow[:,tst_metric] = r[tst_metrica]
            prow[:,val_metric] = r[val_metric]
            prow.seed = 1
            df_images = vcat(df_images, prow)
        end
    end
end

# now filter further
filter!(r->r.modelname in sgad_models_final, df_images)
df_images.modelname[df_images.modelname .== "sgvaegan100"] .= "sgvaegan"

# this generates the overall tables (aggregated by datasets)
df_images = filter(r-> !isnan(r[val_metric]), df_images)
prefix="images_loi"
suffix=""
modelnames = unique(df_images.modelname)
downsample = Dict(zip(modelnames, repeat([DOWNSAMPLE], length(modelnames))))
rts = basic_summary_table(df_images, outdir, prefix=prefix, suffix=suffix,
    downsample=downsample)
save_selection("$(datadir())/evaluation/$(outdir)/$(prefix)_auc_auc_maxmean$(suffix).csv", 
    rts[1], models_alias)

##### LOI images per AC
# this should generate the above tables split by anomaly classes
df_images_target = deepcopy(df_images)
for d in Set(TARGET_DATASETS)
    mask = (df_images_target.dataset .== d)
    df_images_target[mask, :dataset] .= df_images_target[mask, :dataset] .* ":" .* convert_anomaly_class.(df_images_target[mask, :anomaly_class], d)
    df_images_target[mask, :anomaly_class] .= 1 # it has to be > 0, because otherwise we get too many warnings from the aggregate_stats_max_mean
end
df_images_target = filter(r-> !isnan(r.val_auc), df_images_target)

function basic_summary_table_per_ac(df, dir; suffix="", prefix="", downsample=Dict{String, Int}())
    rts = []   
    metric = :auc
    _, rt = sorted_rank(df, aggregate_stats_auto, val_metric, tst_metric, downsample,
        agg_cols=[string(val_metric), string(tst_metric)])
    sorted_models = vcat(["dataset"], [x for x in models_alias if x in names(rt)])
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
    rts
end

prefix="images_loi"
suffix="_per_ac"
modelnames = unique(df_images_target.modelname)
downsample = Dict(zip(modelnames, repeat([DOWNSAMPLE], length(modelnames))))
rts = basic_summary_table_per_ac(df_images_target, outdir, prefix=prefix, suffix=suffix,
    downsample=downsample)
save_selection("$(datadir())/evaluation/$(outdir)/$(prefix)_auc_auc_autoagg$(suffix).csv", 
    rts[1], models_alias)

##### MVTEC
models = ["DeepSVDD", "fAnoGAN", "fmganpy10", "vae", "cgn", "vaegan10", 
    "sgvae", "sgvaegan10", "sgvae_alpha", "sgvaegan_alpha"]
models_alpha = ["sgvae_alpha", "sgvaegan10_alpha"]
MODEL_ALIAS["sgvaegan10"] = "sgvgn"
models_alias = [MODEL_ALIAS[n] for n in models]

# now let's do the same for mvtec results
df_mvtec = load(datadir("evaluation/images_mvtec_eval_all.bson"))[:df];
apply_aliases!(df_mvtec, col="dataset", d=DATASET_ALIAS)
df_mvtec = filter(r->r.modelname in models, df_mvtec)
df_mvtec = filter(r->!(r.dataset in ["grid", "wood"]), df_mvtec)
df_mvtec_nonnan = filter(r-> !isnan(r.val_auc), df_mvtec)

#
df_mvtec_alpha = load(datadir("sgad_alpha_evaluation_kp/images_mvtec_eval.bson"))[:df];
apply_aliases!(df_mvtec_alpha, col="dataset", d=DATASET_ALIAS) # rename
filter!(r->r.modelname in ("sgvae_robreg", "sgvaegan10_robreg"), df_mvtec_alpha)
for m in ["sgvae_", "sgvaegan10_"]
    inds = map(x->occursin(m, x), df_mvtec_alpha.modelname)
    df_mvtec_alpha.modelname[inds] .= m*"alpha"
end
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
