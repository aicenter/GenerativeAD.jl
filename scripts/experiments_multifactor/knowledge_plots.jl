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
include("../evaluation/utils/utils.jl")

outdir = datadir("experiments_multifactor/evaluation_mf_normal/result_tables")
mkpath(outdir)

# load the dfs
df_images = load(datadir("experiments_multifactor/evaluation_mf_normal/images_leave-one-in_eval.bson"))[:df]
df_sgvgna = load(datadir("experiments_multifactor/alpha_evaluation_mf_normal/sgvaegan100_eval.bson"))[:df]
df_sgvaea = load(datadir("experiments_multifactor/alpha_evaluation_mf_normal/sgvae_eval.bson"))[:df]


















sgad_models = ["DeepSVDD", "fAnoGAN", "fmgan", "vae", "cgn", "sgvae", "sgvae_alpha", "sgvae_probreg",
"sgvae_robreg", "sgvae_robreg1", "sgvae_robreg5"]
MODEL_ALIAS["sgvae_robreg"] = "sgvaer"
MODEL_ALIAS["sgvae_robreg1"] = "sgvaer1"
MODEL_ALIAS["sgvae_robreg5"] = "sgvaer5"
sgad_alpha_models = ["sgvae_alpha", "sgvae_probreg", "sgvae_robreg", "sgvae_robreg1", "sgvae_robreg5"]

TARGET_DATASETS = Set(["wmnist"])

function prepare_alpha_df!(df)
    apply_aliases!(df, col="dataset", d=DATASET_ALIAS) # rename
    filter!(r->r.modelname in sgad_models, df)
    df.modelname = "sgvae_alpha"
    df.dataset[df.dataset .== "metal_nut"] .= "nut"
    df["fs_fit_t"] = NaN
    df["fs_eval_t"] = NaN
    df
end

# LOI basic tables
df_images = mf_normal ?
     load(datadir("experiments_multifactor/evaluation_mf_normal/images_leave-one-in_eval.bson"))[:df] :
     load(datadir("experiments_multifactor/evaluation/images_leave-one-in_eval.bson"))[:df];

apply_aliases!(df_images, col="dataset", d=DATASET_ALIAS) # rename
# filter out only the interesting models
df_images = filter(r->r.modelname in sgad_models, df_images)
# now create the column with anomaly factors
parameters = map(x->replace(x, "anomaly_factors" => "anomaly-factors"), df_images.parameters)
df_images[:anomaly_factors] = map(x->parse_savename(x)[2]["anomaly-factors"], parameters) 

# compute number of afs for models
#for model in unique(df_images.modelname)
#    println(model)
#    for af in unique(df_images.anomaly_factors)
#        subdf = filter(r->r.modelname == model && r.anomaly_factors==af, df_images)
#        println("   $af : $(size(subdf,1))")
#    end
#    println("")
#end
df_images_alpha = if mf_normal
    dfs = map(i->load(datadir("experiments_multifactor/alpha_evaluation_mf_normal/images_wildlife_MNIST_eval_ac=$(i).bson"))[:df], 1:10)
    vcat(dfs...)
else
    load(datadir("experiments_multifactor/alpha_evaluation/images_leave-one-in_eval.bson"))[:df];
end;
prepare_alpha_df!(df_images_alpha)
df_images_alpha.anomaly_factors = Meta.parse.(string.(df_images_alpha.anomaly_factors))

# here we want to differentiate sgvae alpha and sgvae alpha with robust logistic regression
df_images_alpha_robreg = filter(r->r.method == "robreg", df_images_alpha)
inds1 = [x.beta for x in df_images_alpha_robreg.parameters] .== 1.0
df_images_alpha_robreg.modelname[inds1] .= "sgvae_robreg1"
df_images_alpha_robreg.modelname[.!inds1] .= "sgvae_robreg5"
filter!(r->r.method!="robreg", df_images_alpha)
df_images_alpha = vcat(df_images_alpha, df_images_alpha_robreg)

# now differentiate them
df_wmnist = df_images
df_wmnist_alpha = df_images_alpha

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
        
        subdf = if !isnothing(df_alpha)
            # now construct a simillar df to be appended to the first one from the alpha df
            kp_nautocols = [string(criterion), replace(string(criterion), "val"=>"tst")]
            subdf_alpha = filter(r->!(isnan(r[criterion])), df_alpha)
            subdf_alpha = subdf_alpha[:,vcat(autocols, kp_nautocols)]
            rename!(subdf_alpha, kp_nautocols[2] => string(tst_metric)) 
            
            # cat the dfs
            vcat(subdf, subdf_alpha)
        else
            subdf
        end

        # now define the agg function
        agg(df,crit) = aggregate_stats_auto(df, crit; agg_cols=nautocols)
        
        if size(subdf, 1) > 0
            df_agg = agg(subdf, criterion)
            
            # some model might be missing
            nr = size(df_agg,2)
            modelnames = df_agg.modelname

            df_agg = add_missing_model!(df_agg, "sgvae_robreg1")
            df_agg = add_missing_model!(df_agg, "sgvae_robreg5")
            
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
title = "wmnist"
round_results = false

# these 3 are in alpha df but not in the normal df
"method", "score_type", "latent_score_type"

# aggragated cols
non_agg_cols = ["modelname","dataset","anomaly_class","phash","parameters","seed","npars",
	"fs_fit_t","fs_eval_t"
]
agg_cols = filter(x->!(x in non_agg_cols), names(df_images))
maxmean_f(df,crit) = aggregate_stats_max_mean(df, crit; agg_cols=[])

# first create the knowledge plot data for the changing level of anomalies
# at different percentages of normal data
cnames = reverse(AUC_METRICS_NAMES)
afs = unique(df_images.anomaly_factors)

level = 100
@suppress_err begin
for af in afs
    #
    df = filter(r->r.anomaly_factors == af,df_wmnist)
    df_alpha = filter(r->r.anomaly_factors == af,df_wmnist_alpha)
    
	criterions = reverse(_prefix_symbol.("val", map(x->x*"_$level",  AUC_METRICS)))
	extended_criterions = vcat(criterions, [val_metric])
	extended_cnames = vcat(["clean"], vcat(cnames, ["\$$(mn)_{val}\$"]))

    ranks_all, metric_means_all = _incremental_rank(df, df_alpha, extended_criterions, tst_metric, 
        non_agg_cols, round_results)
    
    # reorder table on tabular data as there is additional class of models (flows)
    # one can do this manually at the end
    f = joinpath(outdir, "knowledge_plot_multifactor_af-$(af)_$(title)_ano_$(level).csv")
    println("saving to $f")
    CSV.write(f, metric_means_all)
    f = joinpath(outdir, "knowledge_plot_ranks_multifactor_af-$(af)_$(title)_ano_$(level).csv")
    println("saving to $f")
    CSV.write(f, ranks_all)
    ranks_all, metric_means_all
end
end
