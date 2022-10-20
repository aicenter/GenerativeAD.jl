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
using ValueHistories
using GenerativeAD.Evaluation: _prefix_symbol, _get_anomaly_class, _subsample_data
using GenerativeAD.Evaluation: BASE_METRICS, AUC_METRICS
include("../pyutils.jl")
include("./utils.jl")

s = ArgParseSettings()
@add_arg_table! s begin
   "modelname"
        default = "vaegan"
        arg_type = String
        help = "modelname"
    "dataset"
        default = "CIFAR10"
        arg_type = String
        help = "dataset or mvtec category"
    "anomaly_class"
    	default = 0
    	arg_type = Int
    	help = "anomaly class"
    "class_setup"
    	default = 4
    	arg_type = Int
    	help = "how many anomalous classes to use in validation"
    "--force", "-f"
        action = :store_true
        help = "force recomputing of scores"
end
parsed_args = parse_args(ARGS, s)
@unpack modelname, dataset, anomaly_class, class_setup, force = parsed_args
@info "running with these params: $(parsed_args)"
datatype = occursin("MvTEC", dataset) ? "mvtec" : "leave-one-in"
max_ac = (datatype == "mvtec") ? 1 : 10
max_seed = (datatype == "mvtec") ? 5 : 1 
acs = (anomaly_class == 0) ? collect(1:max_ac) : [anomaly_class]
max_seed_perf = 10
classes_val = class_setup
classes_str = (classes_val == 4) ? "" : "_$(classes_val)v$(9-classes_val)"

function experiment(model_id, rf, ac, seed, save_dir, res_dir)
	# 
	outf = joinpath(save_dir, rf)
	if !force && isfile(outf)
		@info "Already present, skipping."
	    return
	end	

	# load the saved data
	data = load(joinpath(res_dir, rf))
	scores_val = data[:val_scores]
	scores_tst = data[:tst_scores]
	y_val = data[:val_labels]
	y_tst = data[:tst_labels]

	# now exclude some data from the validation dataset
	(c_tr, y_tr), (c_val, y_val), (c_tst, y_tst) = original_class_split(dataset, ac, seed=seed)
	# decide the classes that will be used as anomalies
	# the target + the next 4 = normal data
	# the rest is anomalous
	acsn, acsa = divide_classes(ac, classes_val+1)
	val_inds = map(c->c in acsn, c_val);
	
	# this is the final form of the data for the next method
	val_scores = scores_val[val_inds]
	val_y = y_val[val_inds]
	val_c = c_val[val_inds]
	tst_scores = scores_tst
	tst_y = y_tst
	tst_c = c_tst

	return basic_experiment(val_scores, val_y, tst_scores, tst_y, outf, dataset, data, seed, ac)
end	

for ac in acs
	for seed in 1:max_seed
		# make the save dir
		save_dir = datadir("supervised_comparison$(classes_str)/images_$(datatype)/$(modelname)/$(dataset)/ac=$(ac)/seed=$(seed)")
		mkpath(save_dir)
		@info "Saving data to $(save_dir)..."

		# score files
		_res_dir = "experiments/images_$(datatype)/$(modelname)/$(dataset)/ac=$(ac)/seed=$(seed)"
		res_dir = if modelname in ["vae", "DeepSVDD", "fAnoGAN", "fmgan"] && dataset in ["CIFAR10", "SVHN2"]
			"/home/skvarvit/generativead/GenerativeAD.jl/data/" * _res_dir
		else
			datadir(_res_dir)
		end
		rfs = readdir(res_dir)
		rfs = filter(x->!occursin("model", x), rfs)
		model_ids = map(x->Meta.parse(split(split(x, "init_seed=")[2], "_")[1]), rfs)

		# scramble the rest of the models
		n = length(model_ids)
		rand_inds = sample(1:n, n, replace=false)

		# this is what will be iterated over
		final_model_ids = model_ids[rand_inds]
		final_rfs = rfs[rand_inds]
		
		for (model_id, rf) in zip(final_model_ids, final_rfs)
			experiment(model_id, rf, ac, seed, save_dir, res_dir)
		end
		@info "Done."
	end
end
