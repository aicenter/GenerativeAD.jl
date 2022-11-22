# here we do a small scale experiment that compares our composed anomaly score with the one using J(x)
using DrWatson
@quickactivate
using GenerativeAD
using PyCall
using BSON, FileIO, DataFrames
using EvalMetrics
using OrderedCollections
using Suppressor
using StatsBase
using Random
include("../pyutils.jl")
include("../evaluation/utils/ranks.jl")
include("../evaluation/utils/utils.jl")
include("../supervised_comparison/utils.jl")
include("utils.jl")

### first we need to select the best models
using GenerativeAD.Evaluation: MODEL_ALIAS, DATASET_ALIAS, apply_aliases!, convert_anomaly_class
using GenerativeAD.Evaluation: _prefix_symbol, aggregate_stats_mean_max, aggregate_stats_max_mean
using GenerativeAD.Evaluation: _subsample_data
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

modelname = "sgvaegan100_alpha"
base_modelname = "sgvaegan100"
dataset = "SVHN2"
ac = 1
dseed = 40
DOWNSAMPLE = 50
datatype = "leave-one-in"

# load the basic df
df_images = load(datadir("evaluation_kp/images_leave-one-in_eval.bson"))[:df];
# filter out only the interesting models
df_images = filter(r->r.modelname in sgad_models, df_images)
# this generates the overall tables (aggregated by datasets)
df_images = setup_classic_models(df_images)
filter!(r->r.dataset == "svhn2", df_images)

# load the alpha df and edit it slightly
f = "sgad_alpha_evaluation_kp/images_leave-one-in_eval.bson"
df_images_alpha = load(datadir(f))[:df];
df_images_alpha = setup_alpha_models(df_images_alpha)
df_images_alpha = differentiate_beta_1_10(df_images_alpha)
df_images_alpha = differentiate_sgvaegana(df_images_alpha)
filter!(r->r.modelname == modelname, df_images_alpha)
filter!(r->r.dataset == "svhn2", df_images_alpha)

# some setup
df = df_images
df_alpha= df_images_alpha
val_metric = :val_auc_20_100
criterion = val_metric
tst_metric = :tst_auc
non_agg_cols = ["modelname","dataset","anomaly_class","phash","parameters","seed","npars",
    "fs_fit_t","fs_eval_t"]
agg_cols = filter(x->!(x in non_agg_cols), names(df_images))
round_results = false
subdf, _ = glue_classic_and_alpha(df, df_alpha, criterion, tst_metric, 
        replace(string(criterion), "val"=>"tst"), non_agg_cols)
filter!(r->r.modelname == modelname, subdf)
# filter out this model since it gives out Infs in jacodeco
filter!(r->!(occursin("55529190", r.parameters)), subdf)
filter!(r->!(occursin("10954393", r.parameters)), subdf)
filter!(r->!(occursin("83407829", r.parameters) && r.anomaly_class==4), subdf)
filter!(r->!(occursin("70925884", r.parameters) && r.anomaly_class==5), subdf)


# this model seems to be stable and not producing any infs
filter!(r->(occursin("87954753", r.parameters)), subdf) 

modelnames = unique(df.modelname)
downsample = Dict(zip(modelnames, repeat([DOWNSAMPLE], length(modelnames))))
aggdf, acdf = aggregate_stats_max_mean(subdf, criterion; agg_cols=[string(val_metric), string(tst_metric)], 
	downsample=downsample, results_per_ac=true, dseed=1233, topn=1)
df = filter(r->r.modelname == modelname, acdf)
df.tst_auc
df[:cac] = convert_anomaly_class.(df[:, :anomaly_class], "svhn2") 
params = [p[2] for p in parse_savename.(df.parameters)]
df[:model_id] = [p["init_seed"] for p in params] 
sort(df[:,[:model_id, :anomaly_class, :cac, :tst_auc]], :cac)

# now do the alpha experiments with the models
acs = df.anomaly_class
model_ids = df.model_id

# 
i = 6
ac = acs[i]
model_id = model_ids[i]
ps = params[i]
mpath = datadir("sgad_models/images_leave-one-in/sgvaegan100/$(dataset)/ac=$(ac)/seed=1/model_id=$(model_id)")

# load the scores as in the sgad alpha experiment, compute the scores over 10 folds in a single anomaly class
base_beta = 10.0
max_seed_perf = 10
scale = true
init_alpha = [1.0, 1.0, 1.0, 0.1, 0.1, 0.1]
alpha0 = [1, 1, 1, 0, 0, 0]
latent_score_type = "knn"
p = 0.2
p_normal = 0.2
seed = 1

# this now creates the baseline
#val_scores, tst_scores, val_y, tst_y = get_basic_scores(model_id, ac, ps)
#val_auc, tst_auc, alpha = fit_predict_lrnormal(val_scores, tst_scores, val_y, tst_y, 1, p, p_normal)
#_val_scores, _val_y, _ = _subsample_data(p, p_normal, val_y, val_scores; seed=seed)
#_tst_scores, _tst_y, _ = _subsample_data(p, p_normal, tst_y, tst_scores; seed=seed)

# now load the normal data, subsample them and get their jacodeco for them
#orig_data = GenerativeAD.Datasets.load_data(dataset; seed=1, method="leave-one-in", anomaly_class_ind=ac);
#orig_data = GenerativeAD.Datasets.normalize_data(orig_data);
#val_X, val_Y = orig_data[2];
#tst_X, tst_Y = orig_data[3];
#_val_X, _val_Y, _ = _subsample_data(p, p_normal, val_Y, val_X; seed=seed);
#_tst_X, _tst_Y, _ = _subsample_data(p, p_normal, tst_Y, tst_X; seed=seed);

# now we need to load the original model and try to mimick the scores and compute logjacodet from scratch
#model = GenerativeAD.Models.SGVAEGAN(load_sgvaegan_model(mpath, "cuda"))
#scores = StatsBase.predict(model, _tst_X, score_type="discriminator")
#scores = StatsBase.predict(model, _tst_X, score_type="reconstruction", n=10)
#scores = StatsBase.predict(model, _tst_X, score_type="feature_matching", n=10)
# this looks fine - we have a match between the stored scores and the input data
#x = _tst_X[:,:,:,1:10]
#scores = StatsBase.predict(model, x, score_type="log_jacodet")

# create a structure to store the data
function fold_data(model, orig_data, val_scores, tst_scores, val_y, tst_y, p, p_normal, seed, ac, model_id, dataset, ps)
	# subsample the original data
	val_X, val_Y = orig_data[2];
	tst_X, tst_Y = orig_data[3];
	_val_X, _val_Y, _ = _subsample_data(p, p_normal, val_Y, val_X; seed=seed);
	_tst_X, _tst_Y, _ = _subsample_data(p, p_normal, tst_Y, tst_X; seed=seed);
	# subsample the scores
	_val_scores, _val_y, _ = _subsample_data(p, p_normal, val_y, val_scores; seed=seed)
	_tst_scores, _tst_y, _ = _subsample_data(p, p_normal, tst_y, tst_scores; seed=seed)
	val_ljd = StatsBase.predict(model, _val_X, score_type="log_jacodet", batch_size=8)
	tst_ljd = StatsBase.predict(model, _tst_X, score_type="log_jacodet", batch_size=8)
	if any(isinf.(val_ljd))
		@info "validation ljd contains Infs"
	end
	if any(isinf.(tst_ljd))
		@info "test ljd contains Infs"
	end
	ljd = Dict(
		:dataset => dataset,
		:tst_ljd => tst_ljd,
		:val_ljd => val_ljd,
		:tst_scores => _tst_scores,
		:val_scores => _val_scores,
		:tst_y => _tst_y,
		:val_y => _val_y,
		:ac => ac,
		:seed => seed,
		:model_id => model_id,
		:p => p,
		:p_normal => p_normal,
		:params => ps
		)
	return ljd
end

# now loop it
#fd(i) = fold_data(model, val_scores, tst_scores, val_y, tst_y, p, p_normal, i, ac, model_id, dataset, ps)
#jacodata = []
#for i in 1:5
#	push!(jacodata, fd(i))
#	@info "seed $i finished"
#end
#f = datadir("jacodeco/partial_experiment/$(dataset)_$(ac)_all_scores.bson")
#save(f, :jacodata=>jacodata)

# now do it for the rest of the anomaly classes as well
function store_all_folds(dataset, model_id, ac, ps, p, p_normal)
	f = datadir("jacodeco/partial_experiment/$(dataset)_$(ac)_all_scores.bson")
	if isfile(f)
		@info "skipping $f, already exists."
		return
	end
	mpath = datadir("sgad_models/images_leave-one-in/sgvaegan100/$(dataset)/ac=$(ac)/seed=1/model_id=$(model_id)")
	model = GenerativeAD.Models.SGVAEGAN(load_sgvaegan_model(mpath, "cuda"))
	val_scores, tst_scores, val_y, tst_y = get_basic_scores(model_id, ac, ps)
	orig_data = GenerativeAD.Datasets.load_data(dataset; seed=1, method="leave-one-in", anomaly_class_ind=ac);
	orig_data = GenerativeAD.Datasets.normalize_data(orig_data);

	fd(i) = fold_data(model, orig_data, val_scores, tst_scores, val_y, tst_y, p, p_normal, i, ac, model_id, dataset, ps)
	jacodata = []
	for i in 1:5
		push!(jacodata, fd(i))
		@info "seed $i, fold $ac finished"
	end
	
	save(f, :jacodata=>jacodata)
	@info "saved $f"
end

for (model_id, ac, ps) in zip(model_ids, acs, params)
	store_all_folds(dataset, model_id, ac, ps, p, p_normal)
end
