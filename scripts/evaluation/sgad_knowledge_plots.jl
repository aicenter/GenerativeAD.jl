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

include("./utils/ranks.jl")
include("./utils/utils.jl")
outdir = "result_tables"

sgad_models = ["classifier", "DeepSVDD", "fAnoGAN", "fmgan", "fmganpy", "fmganpy10", "vae", "cgn", "cgn_0.2", 
"cgn_0.3", "vaegan", "vaegan10", "sgvaegan", "sgvaegan_0.5", "sgvaegan10", "sgvaegan100", "sgvae", 
"sgvae_alpha", "sgvaegan_alpha"]
sgad_alpha_models = ["classifier", "sgvae_alpha", "sgvaegan_alpha"]
MODEL_ALIAS["cgn_0.2"] = "cgn2"
MODEL_ALIAS["cgn_0.3"] = "cgn3"
MODEL_ALIAS["sgvaegan_0.5"] = "sgvgn05"
MODEL_ALIAS["sgvaegan100"] = "sgvgn100"
MODEL_ALIAS["sgvaegan10_alpha"] = "sgvgn10a"
MODEL_ALIAS["sgvaegan100_alpha"] = "sgvgn100a"
TARGET_DATASETS = Set(["cifar10", "svhn2", "wmnist", "coco"])
round_results = false
DOWNSAMPLE = 50

# LOI basic tables
df_images = load(datadir("evaluation_kp/images_leave-one-in_eval.bson"))[:df];
# filter out only the interesting models
df_images = filter(r->r.modelname in sgad_models, df_images)
# this generates the overall tables (aggregated by datasets)
df_images_target = setup_classic_models(df_images)

# LOI alpha scores
df_images_alpha = load(datadir("sgad_alpha_evaluation_kp/images_leave-one-in_eval.bson"))[:df];
#df_images_alpha = load(datadir("sgad_alpha_evaluation_kp/images_leave-one-in_eval_converted.bson"))[:df];
df_images_alpha_target = setup_alpha_models(df_images_alpha)

# now there is a little bit more differentiation here
df_images_alpha_target = differentiate_beta_1_10(df_images_alpha_target)
df_images_target = differentiate_early_stopping(df_images_target)
df_images_alpha_target = differentiate_sgvaegana(df_images_alpha_target)

# also load the classifier
df_classifier = load(datadir("evaluation_kp/images_leave-one-in_classifier_eval.bson"))[:df]
apply_aliases!(df_classifier, col="dataset", d=DATASET_ALIAS) # rename
df_classifier[:weights_texture] = nothing
df_classifier[:init_alpha] = nothing
df_classifier[:alpha0] = nothing
df_classifier[:fs_fit_t] = nothing
df_classifier[:fs_eval_t] = nothing
df_images_alpha_class = vcat(df_images_alpha_target, df_classifier)

# cgn v 0.3 - differentiate
subdf = filter(r->r.modelname == "cgn_0.3" && occursin("disc_model=conv", r.parameters) &&
    occursin("loss=lin", r.parameters), df_images_target)
subdf.modelname .= "cgn3_lin_conv"
df_images_target = vcat(df_images_target, subdf)
subdf = filter(r->r.modelname == "cgn_0.3" && occursin("disc_model=lin", r.parameters) &&
    occursin("loss=lin", r.parameters), df_images_target)
subdf.modelname .= "cgn3_lin_lin"
df_images_target = vcat(df_images_target, subdf)
subdf = filter(r->r.modelname == "cgn_0.3" && occursin("disc_model=conv", r.parameters) &&
    occursin("loss=log", r.parameters), df_images_target)
subdf.modelname .= "cgn3_log_conv"
df_images_target = vcat(df_images_target, subdf)
subdf = filter(r->r.modelname == "cgn_0.3" && occursin("disc_model=lin", r.parameters) &&
    occursin("loss=lin", r.parameters), df_images_target)
subdf.modelname .= "cgn3_lin_lin"
df_images_target = vcat(df_images_target, subdf)

# now differentiate them by datasets
df_svhn = filter(r->r[:dataset] == "svhn2",df_images_target)
df_svhn_alpha = filter(r->r[:dataset] == "svhn2",df_images_alpha_class)

df_cifar = filter(r->r[:dataset] == "cifar10",df_images_target)
df_cifar_alpha = filter(r->r[:dataset] == "cifar10",df_images_alpha_class)

df_coco = filter(r->r[:dataset] == "coco",df_images_target)
df_coco_alpha = filter(r->r[:dataset] == "coco",df_images_alpha_class)

df_wmnist = filter(r->r[:dataset] == "wmnist",df_images_target)
df_wmnist_alpha = filter(r->r[:dataset] == "wmnist",df_images_alpha_class)

function add_missing_model!(df, modelname)
    nr = size(df,2)
    modelnames = df.modelname
    if !(modelname in modelnames)
        for dataset in unique(df.dataset)
            df = push!(df, vcat(repeat([NaN], nr-2), [dataset, modelname]))
        end
    end
    df
end

function glue_classic_and_alpha(df, df_alpha, val_metric, tst_metric, tst_metrica, non_agg_cols)
    # first separate only the useful columns from the normal eval df
    agg_cols = [string(val_metric), string(tst_metric)]
    subdf = filter(r->!(isnan(r[val_metric]) && !(isnan(r[tst_metric]))), df)
    subdf = subdf[:,vcat(non_agg_cols, agg_cols)] # only use the actually needed columns

    # now construct a simillar df to be appended to the first one from the alpha df
    kp_nautocols = [string(val_metric), string(tst_metrica)]
    subdf_alpha = filter(r->!(isnan(r[val_metric])) && !(isnan(r[tst_metrica])), df_alpha)
    subdf_alpha = subdf_alpha[:,vcat(non_agg_cols, kp_nautocols)]
    rename!(subdf_alpha, kp_nautocols[2] => string(tst_metric)) 

    # now define the agg function and cat it
    modelnames = unique(df.modelname)
    downsample = Dict(zip(modelnames, repeat([DOWNSAMPLE], length(modelnames))))
    agg(df,crit) = aggregate_stats_auto(df, crit; agg_cols=agg_cols, downsample=downsample)
    subdf = vcat(subdf, subdf_alpha)
    return subdf, agg
end

function _incremental_rank(df, df_alpha, criterions, tst_metric, non_agg_cols, round_results)
    ranks, metric_means = [], []
    for criterion in criterions
        subdf, agg = glue_classic_and_alpha(df, df_alpha, criterion, tst_metric, 
                replace(string(criterion), "val"=>"tst"), non_agg_cols)

        if size(subdf, 1) > 0
            df_agg = agg(subdf, criterion)
            
            # some model might be missing
            nr = size(df_agg,2)
            modelnames = df_agg.modelname
            for m in sgad_alpha_models
                if !(m in modelnames)
                    for dataset in unique(df_agg.dataset)
                        df_agg = push!(df_agg, vcat(repeat([NaN], nr-2), [dataset, m]))
                    end
                end
            end
            
            apply_aliases!(df_agg, col="modelname", d=MODEL_RENAME)
            apply_aliases!(df_agg, col="modelname", d=MODEL_ALIAS)
            sort!(df_agg, [:dataset, :modelname])
            rt = rank_table(df_agg, tst_metric; round_results=round_results)
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
titles = ["cifar", "svhn", "coco", "wmnist"]

# aggragated cols
non_agg_cols = ["modelname","dataset","anomaly_class","phash","parameters","seed","npars",
    "fs_fit_t","fs_eval_t"]
agg_cols = filter(x->!(x in non_agg_cols), names(df_images))

# first create the knowledge plot data for the changing level of anomalies
# at different percentages of normal data
cnames = reverse(AUC_METRICS_NAMES)
level = 100
criterions = reverse(_prefix_symbol.("val", map(x->x*"_$level",  AUC_METRICS)))
extended_criterions = vcat(criterions, [val_metric])
extended_cnames = vcat(["clean"], vcat(cnames, ["\$$(mn)_{val}\$"]))

function produce_tables()
    DOWNSAMPLE = 50
    ranks_dfs = @suppress_err begin
        ranks_dfs = map(enumerate(zip(titles,
                [
                    (df_cifar, df_cifar_alpha),
                    (df_svhn, df_svhn_alpha),
                    (df_coco, df_coco_alpha),
                    (df_wmnist, df_wmnist_alpha)]))) do (i, (title, (df, df_alpha)))

            ranks, metric_means = _incremental_rank(df, df_alpha, extended_criterions, tst_metric, 
                non_agg_cols, round_results)
            
            # reorder table on tabular data as there is additional class of models (flows)
            # one can do this manually at the end
            f = joinpath(datadir(), "evaluation", outdir, "kp_v3_$(title)_downsampled.csv")
            println("saving to $f")
            CSV.write(f, metric_means)
            ranks, metric_means
        end
        ranks_dfs
    end
    ranks_dfs
end
kplots = produce_tables()

# now do it again, this time per anomaly class and only for certain metrics
val_metric = :val_auc_100_100
tst_metrica = :tst_auc_100_100 
tst_metric = :tst_auc
all_df,_ = glue_classic_and_alpha(df_images_target, df_images_alpha_class, val_metric, tst_metric, 
                tst_metrica, non_agg_cols)

# downsampling
DOWNSAMPLE = 50
modelnames = unique(all_df.modelname)
# rename the models
for model in modelnames
    all_df.modelname[all_df.modelname .== model] .= get(MODEL_ALIAS, model, model)
end
downsample = Dict(zip(modelnames, repeat([DOWNSAMPLE], length(modelnames))))
resa, subresa = aggregate_stats_max_mean(all_df, val_metric; results_per_ac=true, 
    agg_cols=[string(val_metric), string(tst_metric)], downsample=downsample)
f = joinpath(datadir(), "evaluation", outdir, "kp_v3_table_per_ac.csv")
CSV.write(f, subresa[:,["modelname", "dataset", "anomaly_class", "parameters", "seed",
    "val_auc_100_100", "tst_auc"]])
