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
    "datatype"
        default = "leave-one-in"
        arg_type = String
        help = "leave-one-in or mvtec"
    "latent_score_type"
        arg_type = String
        help = "normal, kld, knn or normal_logpx"
        default = "normal"
    "--force", "-f"
        action = :store_true
        help = "force recomputing of scores"
end
parsed_args = parse_args(ARGS, s)
@unpack modelname, dataset, datatype, latent_score_type, force = parsed_args
max_ac = (datatype == "mvtec") ? 1 : 10
max_seed = (datatype == "mvtec") ? 5 : 1 

score_type = "logpx"
device = "cpu"
method = "original"
max_seed_perf = 10

function basic_stats(labels, scores)
	try
		roc = EvalMetrics.roccurve(labels, scores)
		auc = EvalMetrics.auc_trapezoidal(roc...)
		prc = EvalMetrics.prcurve(labels, scores)
		auprc = EvalMetrics.auc_trapezoidal(prc...)

		t5 = EvalMetrics.threshold_at_fpr(labels, scores, 0.05)
		cm5 = ConfusionMatrix(labels, scores, t5)
		tpr5 = EvalMetrics.true_positive_rate(cm5)
		f5 = EvalMetrics.f1_score(cm5)

		return auc, auprc, tpr5, f5
	catch e
		if isa(e, ArgumentError)
			return NaN, NaN, NaN, NaN
		else
			rethrow(e)
		end
	end
end

auc_val(labels, scores) = EvalMetrics.auc_trapezoidal(EvalMetrics.roccurve(labels, scores)...)

function perf_at_p_new(p, p_normal, val_scores, val_y, tst_scores, tst_y; seed=nothing)
	scores, labels, _ = _subsample_data(p, p_normal, val_y, val_scores; seed=seed)
	# if there are no positive samples return NaNs
	if sum(labels) == 0
		val_auc = NaN
		tst_auc = auc_val(tst_y, tst_scores[:,1])
	# if top samples are only positive
	# we cannot train alphas
	# therefore we return the default val performance
	elseif sum(labels) == length(labels) 
		val_auc = NaN
		tst_auc = auc_val(tst_y, tst_scores[:,1])
	# if they are not only positive, then we train alphas and use them to compute 
	# new scores - auc vals on the partial validation and full test dataset
	else
		try
			lr = LogReg()
			fit!(lr, scores, labels)
			val_probs = predict(lr, scores)
			tst_probs = predict(lr, tst_scores)
			val_auc = auc_val(labels, val_probs)
			tst_auc = auc_val(tst_y, tst_probs)
		catch e
			if isa(e, LoadError) || isa(e, ArgumentError)
				val_prec = NaN
				val_auc = NaN
				tst_auc = auc_val(tst_y, tst_scores[:,1])
			else
				rethrow(e)
			end
		end
	end
	return val_auc, tst_auc
end	

function perf_at_p_agg(args...)
	results = [perf_at_p_new(args...;seed=seed) for seed in 1:max_seed_perf]
	return mean([x[1] for x in results]), mean([x[2] for x in results])
end

# this is for fitting the logistic regression
mutable struct LogReg
	alpha
end

LogReg() = LogReg(nothing)

function fit!(lr::LogReg, X, y)
	py"""
from sgad.sgvae.utils import logreg_fit

def fit(X,y):
	alpha, _ = logreg_fit(X, 1-y)
	return alpha
	"""
	lr.alpha = py"fit"(X, y)
end

function predict(lr::LogReg, X)
	py"""
from sgad.sgvae.utils import logreg_prob

def predict(X, alpha):
	return logreg_prob(X, alpha)
	"""
	return py"predict"(X, lr.alpha)
end

for ac in 1:max_ac
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
		save_dir = datadir("sgad_alpha_evaluation_kp/images_$(datatype)/$(modelname)/$(dataset)/ac=$(ac)/seed=$(seed)")
		mkpath(save_dir)
		@info "Saving data to $(save_dir)..."

		# top score files
		res_dir = datadir("experiments/images_$(datatype)/$(modelname)/$(dataset)/ac=$(ac)/seed=$(seed)")
		rfs = readdir(res_dir)
		rfs = filter(x->occursin(score_type, x), rfs)

		for (model_id, lf) in zip(model_ids, lfs)
			outf = joinpath(save_dir, split(lf, ".")[1] * "_method=$(method).bson")
			if !force && isfile(outf)
				continue
			end	

			# load the saved scores
			ldata = load(joinpath(latent_dir, lf))
			rf = filter(x->occursin("$(model_id)", x), rfs)
			if length(rf) < 1
				@info "Something is wrong, original score file for $lf not found"
				continue
			end
			rf = rf[1]
			rdata = load(joinpath(res_dir, rf))

			# prepare the data
			if isnan(ldata[:val_scores][1])
				continue
			end
			val_scores = cat(rdata[:val_scores], transpose(ldata[:val_scores]), dims=2);
			tst_scores = cat(rdata[:tst_scores], transpose(ldata[:tst_scores]), dims=2);
			tr_y = ldata[:tr_labels];
			val_y = ldata[:val_labels];
			tst_y = ldata[:tst_labels];

			# setup params
			parameters = ldata[:parameters]
			add_params = split(split(lf, "score")[1], "model_id=$(model_id)")[2]
			param_string = "latent_score_type=$(latent_score_type)" * add_params * split(rf, ".bson")[1]
			
			res_df = @suppress begin
				# prepare the result dataframe
				res_df = OrderedDict()
				res_df["modelname"] = modelname
				res_df["dataset"] = dataset
				res_df["phash"] = GenerativeAD.Evaluation.hash(parameters)
				res_df["parameters"] = param_string
				res_df["fit_t"] = rdata[:fit_t]
				res_df["tr_eval_t"] = ldata[:tr_eval_t] + rdata[:tr_eval_t]
				res_df["val_eval_t"] = ldata[:val_eval_t] + rdata[:val_eval_t]
				res_df["tst_eval_t"] = ldata[:tst_eval_t] + rdata[:tst_eval_t]
				res_df["seed"] = seed
				res_df["npars"] = rdata[:npars]
				res_df["anomaly_class"] = ac
				res_df["method"] = method
				res_df["score_type"] = score_type
				res_df["latent_score_type"] = latent_score_type

				# fit the logistic regression - first on all the validation data
				# first, filter out NaNs and Infs
				inds = vec(mapslices(r->!any(r.==Inf), val_scores, dims=2))
				val_scores = val_scores[inds, :]
				val_y = val_y[inds]
				inds = vec(mapslices(r->!any(isnan.(r)), val_scores, dims=2))
				val_scores = val_scores[inds, :]
				val_y = val_y[inds]

				lr = LogReg()
				fit!(lr, val_scores, val_y)
				val_probs = predict(lr, val_scores)
				tst_probs = predict(lr, tst_scores)

				# now fill in the values
				res_df["val_auc"], res_df["val_auprc"], res_df["val_tpr_5"], res_df["val_f1_5"] = 
					basic_stats(val_y, val_probs)
				res_df["tst_auc"], res_df["tst_auprc"], res_df["tst_tpr_5"], res_df["tst_f1_5"] = 
					basic_stats(tst_y, tst_probs)

				# then do the same on a small section of the data
				auc_ano_100 = [perf_at_p_agg(p/100, 1.0, val_scores, val_y, tst_scores, tst_y) 
					for p in [100.0, 50.0, 20.0, 10.0, 5.0, 2.0, 1.0, 0.5, 0.2, 0.1]]
				for (k,v) in zip(map(x->x * "_100", AUC_METRICS), auc_ano_100)
					res_df[k] = v
				end

				auc_ano_50 = [perf_at_p_agg(p/100, 0.5, val_scores, val_y, tst_scores, tst_y) 
					for p in [100.0, 50.0, 20.0, 10.0, 5.0, 2.0, 1.0, 0.5, 0.2, 0.1]]
				for (k,v) in zip(map(x->x * "_50", AUC_METRICS), auc_ano_50)
					res_df[k] = v
				end

				auc_ano_10 = [perf_at_p_agg(p/100, 0.1, val_scores, val_y, tst_scores, tst_y) 
					for p in [100.0, 50.0, 20.0, 10.0, 5.0, 2.0, 1.0, 0.5, 0.2, 0.1]]
				for (k,v) in zip(map(x->x * "_10", AUC_METRICS), auc_ano_10)
					res_df[k] = v
				end

				prop_ps = [100, 50, 20, 10, 5, 2, 1]
				auc_prop_100 = [perf_at_p_agg(1.0, p/100, val_scores, val_y, tst_scores, tst_y) 
					for p in prop_ps]
				for (k,v) in zip(map(x-> "auc_100_$(x)", prop_ps), auc_prop_100)
					res_df[k] = v
				end
				res_df
			end
			
			# then save it
			res_df = DataFrame(res_df)
			save(outf, Dict(:df => res_df))
			#@info "Saved $outf."
		end
		@info "Done."
	end
end
