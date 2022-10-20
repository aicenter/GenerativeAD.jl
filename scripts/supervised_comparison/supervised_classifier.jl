# this is meant to compute the performance of a supervised classifier that is trained on
# only 4 classes of anomalies, but tested on 9
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
include("./classifier.jl")

s = ArgParseSettings()
@add_arg_table! s begin
    "dataset"
        default = "wildlife_MNIST"
        arg_type = String
        help = "dataset or mvtec category"
    "anomaly_class"
    	default = 0
    	arg_type = Int
    	help = "anomaly class"
    "class_setup"
    	default = 4
    	arg_type = Int
    	help = "how many anomalous classes to use in training"
end
parsed_args = parse_args(ARGS, s)
@unpack dataset, anomaly_class, class_setup = parsed_args
datatype = occursin("MvTec", dataset) ? "mvtec" : "leave-one-in"
max_ac = (datatype == "mvtec") ? 1 : 10
max_seed = (datatype == "mvtec") ? 5 : 1 
acs = (anomaly_class == 0) ? collect(1:max_ac) : [anomaly_class]
modelname = "classifier"
niters = 1000
max_seed_perf = 10
classes_val = class_setup
classes_str = (classes_val == 4) ? "" : "_$(classes_val)v$(9-classes_val)"

function experiment(dataset, ac, seed, save_dir)
	# get the data
	(tr_x, tr_y), (tst_x, tst_y) = basic_classifier_inputs(dataset, ac, seed, classes_val)

	# get classifier parameters
	parameters = sample_params()
	parameters = GenerativeAD.edit_params(tr_x, parameters)

	# prepare the save file
	outf = prepare_savefile(save_dir, parameters)
	(outf == "") ? (return nothing) : nothing

	res_df = OrderedDict()
	res_df["modelname"] = modelname
	res_df["dataset"] = dataset
	res_df["phash"] = GenerativeAD.Evaluation.hash(parameters)
	res_df["parameters"] = "_"*extended_savename(parameters)
	res_df["fit_t"] = NaN
	res_df["tr_eval_t"] = NaN
	res_df["val_eval_t"] = NaN
	res_df["tst_eval_t"] = NaN
	res_df["seed"] = seed
	res_df["npars"] = NaN
	res_df["anomaly_class"] = ac
	res_df["method"] = nothing
	res_df["score_type"] = nothing
	res_df["latent_score_type"] = nothing

	# first fit the classifier on the full data
	model, history, tr_probs, tst_probs = fit_classifier(tr_x, tr_y, tst_x, tst_y, parameters, niters)
	
	# now fill in the values
	res_df["val_auc"], res_df["val_auprc"], res_df["val_tpr_5"], res_df["val_f1_5"] = 
		basic_stats(tr_y, tr_probs)
	res_df["tst_auc"], res_df["tst_auprc"], res_df["tst_tpr_5"], res_df["tst_f1_5"] = 
		basic_stats(tst_y, tst_probs)

	# then do the same on a small section of the data
	ps = [100.0, 50.0, 20.0, 10.0, 5.0, 2.0, 1.0, 0.5, 0.2, 0.1] ./ 100
	auc_ano_100 = [perf_at_p_agg(p, 1.0, tr_x, tr_y, tst_x, tst_y, parameters, niters) for p in ps]
	for (k,v) in zip(map(x->x * "_100", AUC_METRICS), auc_ano_100)
		res_df["val_"*k] = v[1]
		res_df["tst_"*k] = v[2]
	end

	# then save it
	res_df = DataFrame(res_df)
	save(outf, Dict(:df => res_df))
	res_df
end	

for ac in acs
	for seed in 1:max_seed
		save_dir = datadir("supervised_comparison$(classes_str)/images_$(datatype)/$(modelname)/$(dataset)/ac=$(ac)/seed=$(seed)")
		mkpath(save_dir)
		@info "Saving data to $(save_dir)..."

		df = experiment(dataset, ac, seed, save_dir)
		@info "Done."
	end
end
