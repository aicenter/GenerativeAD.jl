# this is meant to recompute the scores using the latent scores and alpha coefficients for a model
using DrWatson
@quickactivate
using GenerativeAD
using PyCall
using BSON, FileIO, DataFrames
using EvalMetrics
using OrderedCollections
using ArgParse
using Suppressor
using StatsBase
using Random
using GenerativeAD.Evaluation: _prefix_symbol, _get_anomaly_class, _subsample_data
using GenerativeAD.Evaluation: BASE_METRICS, AUC_METRICS
include("../pyutils.jl")
include("./utils.jl")

s = ArgParseSettings()
@add_arg_table! s begin
   "modelname"
        default = "sgvae"
        arg_type = String
        help = "modelname"
    "dataset"
        default = "wildlife_MNIST"
        arg_type = String
        help = "dataset or mvtec category"
    "latent_score_type"
        arg_type = String
        help = "normal, kld, knn or normal_logpx"
        default = "knn"
    "anomaly_class"
    	default = 0
    	arg_type = Int
    	help = "anomaly class"
    "base_beta"
    	default = 1.0
    	arg_type = Float64
    	help = "base beta for robust logistic regression"
    "--force", "-f"
        action = :store_true
        help = "force recomputing of scores"
end
parsed_args = parse_args(ARGS, s)
@unpack modelname, dataset, latent_score_type, anomaly_class, base_beta, force = parsed_args
datatype = occursin("MvTEC", dataset) ? "mvtec" : "leave-one-in"
max_ac = (datatype == "mvtec") ? 1 : 10
max_seed = (datatype == "mvtec") ? 5 : 1 
acs = (anomaly_class == 0) ? collect(1:max_ac) : [anomaly_class]

method = "robreg"
score_type = "logpx"
device = "cpu"
max_seed_perf = 10
scale = true
init_alpha = [1.0, 0.1, 0.1, 0.1]

function experiment(model_id, lf, ac, seed, latent_dir, save_dir, res_dir, rfs)
	# 
	outf = prepare_savefile(save_dir, lf, base_beta, method)
	(outf == "") ? (return nothing) : nothing

	# load base and latent scores
	scores_val, scores_tst, y_val, y_tst, ldata, rdata = load_scores(model_id, lf, latent_dir, rfs, res_dir)

	# now exclude some data from the validation dataset
	(c_tr, y_tr), (c_val, y_val), (c_tst, y_tst) = original_class_split(dataset, ac, seed=seed)
	# decide the classes that will be used as anomalies
	# the target + the next 4 = normal data
	# the rest is anomalous
	acsn, acsa = divide_classes(ac)
	val_inds = map(c->c in acsn, c_val);
	tst_inds = map(c->c in acsn, c_tst);

	# this is the final form of the data for the next method
	val_scores = scores_val[val_inds,:]
	val_y = y_val[val_inds]
	val_c = c_val[val_inds]
	tst_scores = scores_tst
	tst_y = y_tst
	tst_c = c_tst

	return basic_experiment(val_scores, val_y, tst_scores, tst_y, outf, base_beta, init_alpha, 
		scale, dataset, rdata, ldata, seed, ac, method, score_type, latent_score_type)
end	

# this is the part where we load the best models
bestf = datadir("sgad_alpha_evaluation_kp/best_models_$(datatype).bson")
best_models = load(bestf)

for ac in acs
	for seed in 1:max_seed
		# we will go over the models that have the latent scores computed - for them we can be sure that 
		# we have all we need
		# we actually don't even need to load the models themselves, just the original (logpx) scores
		# and the latent scores and a logistic regression solver from scikit
		latent_dir = datadir("sgad_latent_scores/images_$(datatype)/$(modelname)/$(dataset)/ac=$(ac)/seed=$(seed)")
		lfs = readdir(latent_dir)
		ltypes = map(lf->split(split(lf, "score=")[2], ".")[1], lfs)
		lfs = lfs[ltypes .== latent_score_type]
		model_ids = map(x->Meta.parse(split(split(x, "=")[2], "_")[1]), lfs)

		# make the save dir
		save_dir = datadir("supervised_comparison/images_$(datatype)/$(modelname)/$(dataset)/ac=$(ac)/seed=$(seed)")
		mkpath(save_dir)
		@info "Saving data to $(save_dir)..."

		# top score files
		res_dir = datadir("experiments/images_$(datatype)/$(modelname)/$(dataset)/ac=$(ac)/seed=$(seed)")
		rfs = readdir(res_dir)
		rfs = filter(x->occursin(score_type, x), rfs)

		# this is where we select the files of best models
		# now add the best models to the mix
		inds = (best_models[:anomaly_class] .== ac) .& (best_models[:seed] .== seed) .& 
			(best_models[:dataset] .== dataset)
		best_params = best_models[:parameters][inds]

		# from these params extract the correct model_ids and lfs
		parsed_params = map(x->parse_savename("s_$x")[2], best_params)
		best_model_ids = [x["init_seed"] for x in parsed_params]
		best_lfs = map(x->get_latent_file(x, lfs), parsed_params)

		# use only those that are not nothing - in agreement with the latent_score_type
		used_inds = .!map(isnothing, best_lfs)

		# also, scramble the rest of the models
		n = length(model_ids)
		rand_inds = sample(1:n, n, replace=false)

		# this is what will be iterated over
		final_model_ids = vcat(best_model_ids[used_inds], model_ids[rand_inds])
		final_lfs = vcat(best_lfs[used_inds], lfs[rand_inds])
		
		for (model_id, lf) in zip(final_model_ids, final_lfs)
			experiment(model_id, lf, ac, seed, latent_dir, save_dir, res_dir, rfs)
		end
		@info "Done."
	end
end
