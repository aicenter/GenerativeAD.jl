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

sgad_models = ["DeepSVDD", "fAnoGAN", "fmgan", "fmganpy", "vae", "cgn", "vaegan", "sgvaegan", "sgvae", 
    "sgvae_alpha", "sgvaegan_alpha"]
sgad_alpha_models = ["sgvae_alpha", "sgvaegan_alpha"]
MODEL_ALIAS["sgvaegan10_alpha"] = "sgvgn10a"
TARGET_DATASETS = Set(["cifar10", "svhn2", "wmnist", "coco"])
round_results = false
DOWNSAMPLE = 150
# downsample in aggregate_stats_auto and aggregate_stats_max_mean

function prepare_alpha_df!(df)
    apply_aliases!(df, col="dataset", d=DATASET_ALIAS) # rename
    df.dataset[df.dataset .== "metal_nut"] .= "nut"
    df["fs_fit_t"] = NaN
    df["fs_eval_t"] = NaN
    df
end

# LOI basic tables
df_images = load(datadir("evaluation_kp/images_leave-one-in_eval.bson"))[:df];
apply_aliases!(df_images, col="dataset", d=DATASET_ALIAS) # rename
# filter out only the interesting models
df_images = filter(r->r.modelname in sgad_models, df_images)
# this generates the overall tables (aggregated by datasets)
df_images_target, _ = _split_image_datasets(df_images, TARGET_DATASETS);
df_images_target = filter(r->r.modelname != "sgvae_alpha", df_images_target);

# LOI alpha scores
df_images_alpha = load(datadir("sgad_alpha_evaluation_kp/images_leave-one-in_eval.bson"))[:df];
#df_images_alpha = load(datadir("sgad_alpha_evaluation_kp/images_leave-one-in_eval_converted.bson"))[:df];
filter!(r->occursin("_robreg", r.modelname), df_images_alpha)
filter!(r->get(parse_savename(r.parameters)[2], "beta", 1.0) in [1.0, 10.0], df_images_alpha)
for model in ["sgvae_", "sgvaegan_", "sgvaegan10_"]
    df_images_alpha.modelname[map(r->occursin(model, r.modelname), eachrow(df_images_alpha))] .= model*"alpha"
end
prepare_alpha_df!(df_images_alpha)
df_images_alpha_target, _ = _split_image_datasets(df_images_alpha, TARGET_DATASETS);

# now add the models we are interested in
# sgvaeganalpha - beta=1/10
subdfa = filter(r->r.modelname == "sgvaegan_alpha", df_images_alpha_target)
parametersa = map(x->parse_savename(x)[2], subdfa.parameters)
subdfa.modelname[[x["beta"] for x in parametersa] .== 1.0] .= "sgvaegan_alpha_1"
subdfa.modelname[[x["beta"] for x in parametersa] .== 10.0] .= "sgvaegan_alpha_10"
df_images_alpha_target = vcat(df_images_alpha_target, subdfa)
MODEL_ALIAS["sgvaegan_alpha_1"] = "sgvgna1"
MODEL_ALIAS["sgvaegan_alpha_10"] = "sgvgna10"

# sgvaegan/vaegan/fmganpy - 1000 or 10 early stopping anomalies
subdf = filter(r->r.modelname in ["sgvaegan", "vaegan", "fmganpy"], df_images_target)
parameters = map(x->parse_savename(x)[2], subdf.parameters)
vs = [get(x, "version", 0.3) for x in parameters]
subdf.modelname[vs .== 0.3] .= subdf.modelname[vs .== 0.3] .* "_0.3"
subdf.modelname[vs .== 0.4] = subdf.modelname[vs .== 0.4] .* "_0.4"
df_images_target = vcat(df_images_target, subdf)
MODEL_ALIAS["sgvaegan_0.3"] = "sgvgn1000"
MODEL_ALIAS["vaegan_0.3"] = "vgn1000"
MODEL_ALIAS["fmganpy_0.3"] = "fmgn1000"
MODEL_ALIAS["sgvaegan_0.4"] = "sgvgn10"
MODEL_ALIAS["vaegan_0.4"] = "vgn10"
MODEL_ALIAS[23645901"fmganpy_0.4"] = "fmgn10"

subdfa = filter(r->r.modelname == "sgvaegan_alpha", df_images_alpha_target)
parametersa = map(x->parse_savename(x)[2], subdfa.parameters)
subdfa.modelname[[get(x, "version", 0.3) for x in parametersa] .== 0.3] .= "sgvaegan_alpha_0.3"
subdfa.modelname[[get(x, "version", 0.3) for x in parametersa] .== 0.4] .= "sgvaegan_alpha_0.4"


#########šššš
subdf = filter(r->r-modelname == "sgvaegan_0.4")

outf = datadir("sgad_alpha_evaluation_kp/best_models_orig_leave-one-in.bson")
d = load(outf)
i = 1540
for k in keys(d)
    println(d[k][i])
end

1
wildlife_MNIST
sgvaegan
activation=leakyrelu_batch_size=16_fm_alpha=1000_fm_depth=6_h_channels=32_img_dim=32_init_gain=0.03_init_seed=23645901_lr=0.000398_n_layers=4_optimizer=adam_score=discriminator_version=0.4_weight_binary=100_weight_mask=0.05_z_dim=128
1

ps = parameters[[x["init_seed"] for x in parameters] .== 23645901]
ps = parametersa[[x["init_seed"] for x in parametersa] .== 23645901]
#########

# now differentiate them
df_svhn = filter(r->r[:dataset] == "svhn2",df_images_target)
df_svhn_alpha = filter(r->r[:dataset] == "svhn2",df_images_alpha_target)

df_cifar = filter(r->r[:dataset] == "cifar10",df_images_target)
df_cifar_alpha = filter(r->r[:dataset] == "cifar10",df_images_alpha_target)

df_coco = filter(r->r[:dataset] == "coco",df_images_target)
df_coco_alpha = filter(r->r[:dataset] == "coco",df_images_alpha_target)

df_wmnist = filter(r->r[:dataset] == "wmnist",df_images_target)
df_wmnist_alpha = filter(r->r[:dataset] == "wmnist",df_images_alpha_target)

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

function _incremental_rank(df, df_alpha, criterions, tst_metric, non_agg_cols, round_results)
    ranks, metric_means = [], []
    for criterion in criterions
        
        # first separate only the useful columns from the normal eval df
        autocols = non_agg_cols
        nautocols = [string(criterion), string(tst_metric)]
        subdf = filter(r->!(isnan(r[criterion])), df)
        subdf = subdf[:,vcat(autocols, nautocols)] # only use the actually needed columns
        
        # now construct a simillar df to be appended to the first one from the alpha df
        kp_nautocols = [string(criterion), replace(string(criterion), "val"=>"tst")]
        subdf_alpha = filter(r->!(isnan(r[criterion])), df_alpha)
        subdf_alpha = subdf_alpha[:,vcat(autocols, kp_nautocols)]
        rename!(subdf_alpha, kp_nautocols[2] => string(tst_metric)) 
        
        # now define the agg function and cat it
        modelnames = unique(df.modelname)
        downsample = Dict(zip(modelnames, repeat([DOWNSAMPLE], length(modelnames))))
        agg(df,crit) = aggregate_stats_auto(df, crit; agg_cols=nautocols, downsample=downsample)
        subdf = vcat(subdf, subdf_alpha)

        if size(subdf, 1) > 0
            df_agg = agg(subdf, criterion)
            
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

@suppress_err begin
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
	    f = joinpath(datadir(), "evaluation", outdir, "kp_complex_$(title).csv")
	    println("saving to $f")
	    CSV.write(f, metric_means)
	    ranks, metric_means
	end
end

DOWNSAMPLE = 50
@suppress_err begin
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
        f = joinpath(datadir(), "evaluation", outdir, "kp_complex_$(title)_downsampled.csv")
        println("saving to $f")
        CSV.write(f, metric_means)
        ranks, metric_means
    end
end
