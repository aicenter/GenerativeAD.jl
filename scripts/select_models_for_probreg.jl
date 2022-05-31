using DrWatson
@quickactivate
using GenerativeAD
using FileIO, BSON, DataFrames
using StatsBase
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

# setup
modelname = "sgvae_alpha"
sgad_models = ["sgvae","sgvae_alpha"]
n_models = 10

# functions
function prepare_alpha_df!(df)
    filter!(r->r.modelname in sgad_models, df)
    df.modelname = "sgvae_alpha"
    df.dataset[df.dataset .== "metal_nut"] .= "nut"
    df["fs_fit_t"] = NaN
    df["fs_eval_t"] = NaN
    df
end

function best_models(df, modelnames, datasets, seeds, acs, criterions, latent_score_types)
	# setup a save df
	outd = Dict()
	outd[:seed] = []
	outd[:anomaly_class] = []
	outd[:dataset] = []
	outd[:modelname] = []
	outd[:parameters] = []
	outd[:latent_score_type] = []

	for modelname in modelnames
		for latent_score_type in latent_score_types
			for dataset in datasets
				for seed in seeds
					for ac in acs
						for crit in criterions
							val_crit, tst_crit = crit;
							# now select the best hyperparams
							subdf = filter(r->r.modelname == modelname && r.dataset == dataset && 
								r.seed == seed && r.anomaly_class == ac && !isnan(r[val_crit]) &&
								r.latent_score_type == latent_score_type, df);
							if size(subdf, 1) > 0
								imaxs = sortperm(subdf[val_crit], rev=true);
								# write it into the dict
								n_max = minimum(n_models, size(subdf,1))
								for imax in imaxs[1:n_max]
									bestdf = subdf[imax,:]
									push!(outd[:seed], seed)
									push!(outd[:anomaly_class], ac)
									push!(outd[:dataset], dataset)
									push!(outd[:modelname], modelname)
									push!(outd[:parameters], bestdf.parameters)
									push!(outd[:latent_score_type], latent_score_type)
								end
							end
						end
					end
				end
			end
		end
	end
	return outd
end

# criterions
criterions = (
	(:val_auc, :tst_auc),
	(:val_auc_100_100, :tst_auc_100_100),
	(:val_auc_50_100, :tst_auc_50_100),
	(:val_auc_20_100, :tst_auc_20_100),
	(:val_auc_10_100, :tst_auc_10_100),
	(:val_auc_5_100, :tst_auc_5_100),
	(:val_auc_2_100, :tst_auc_2_100),
	(:val_auc_1_100, :tst_auc_1_100),
	(:val_auc_05_100, :tst_auc_05_100),
	(:val_auc_02_100, :tst_auc_02_100),
	(:val_auc_01_100, :tst_auc_01_100)
	)

# LOI
df_images_alpha = load(datadir("sgad_alpha_evaluation_kp/images_leave-one-in_eval.bson"))[:df];
prepare_alpha_df!(df_images_alpha)
modelnames = unique(df_images_alpha.modelname) 
datasets = unique(df_images_alpha.dataset)
latent_score_types = unique(df_images_alpha.latent_score_type)
seeds = 1:1
acs = 1:10

outf = datadir("sgad_alpha_evaluation_kp/best_models_leave-one-in.bson") 
outd = best_models(df_images_alpha, modelnames, datasets, seeds, acs, criterions, latent_score_types)
save(outf, outd)

# mvtec
df_images_mvtec = load(datadir("sgad_alpha_evaluation_kp/images_mvtec_eval.bson"))[:df];
prepare_alpha_df!(df_images_mvtec)
modelnames = unique(df_images_mvtec.modelname) 
datasets = unique(df_images_mvtec.dataset)
latent_score_types = unique(df_images_mvtec.latent_score_type)
seeds = 1:5
acs = 1:1

outf = datadir("sgad_alpha_evaluation_kp/best_models_mvtec.bson") 
outd = best_models(df_images_mvtec, modelnames, datasets, seeds, acs, criterions, latent_score_types)
save(outf, outd)
