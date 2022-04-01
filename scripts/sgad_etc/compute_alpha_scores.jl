# this is meant to recompute the scores using the latent scores and alpha coefficients for a model
using DrWatson
@quickactivate
using GenerativeAD
using PyCall
using BSON, FileIO, DataFrames
using EvalMetrics
using OrderedCollections

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
        help = "normal, kld or normal_logpx"
        default = "normal"
    "device"
        arg_type = String
        help = "cpu or cuda"
        default = "cpu"
    "--force", "-f"
        action = :store_true
        help = "force recomputing of scores"
end
parsed_args = parse_args(ARGS, s)
@unpack modelname, dataset, datatype, latent_score_type, device, force = parsed_args
max_ac = (datatype == "mvtec") ? 1 : 10
max_seed = (datatype == "mvtec") ? 5 : 1 

score_type = "logpx"
device = "cpu"
method = "original"

		# how do the original eval files look?
#		ef = datadir("sgad_alpha_evaluation/prototype.bson")
#		edata = load(ef)[:df]


"""
	top_scores_at_p(p, scores)

Assumes that the scores are of shape (nsamples, nfeatures).
Returns the 100*p% of samples based on the values in the first column 
which is supposed to be the top score. 
"""
function top_samples_at_p(p, scores, labels)
	pN = floor(Int, p*size(scores,1))
	inds = sortperm(scores[:,1],rev=true)[1:pN]
	scores[inds,:], labels[inds], inds
end

function basic_stats(labels, scores)
	roc = EvalMetrics.roccurve(labels, scores)
	auc = EvalMetrics.auc_trapezoidal(roc...)
	prc = EvalMetrics.prcurve(labels, scores)
	auprc = EvalMetrics.auc_trapezoidal(prc...)

	t5 = EvalMetrics.threshold_at_fpr(labels, scores, 0.05)
	cm5 = ConfusionMatrix(labels, scores, t5)
	tpr5 = EvalMetrics.true_positive_rate(cm5)
	f5 = EvalMetrics.f1_score(cm5)

	return auc, auprc, tpr5, f5
end

auc_val(labels, scores) = EvalMetrics.auc_trapezoidal(EvalMetrics.roccurve(labels, scores)...)

# TODO also change this method to something else, e.g.
function perf_at_p_original(p, val_scores, val_y, tst_scores, tst_y)
	scores, labels, _ = top_samples_at_p(p, val_scores, val_y)
	# if top samples are only positive
	# we cannot train alphas
	# therefore we return the default val performance
	if sum(labels) == length(labels) 
		val_prec = 1.0
		val_auc = 1.0
		tst_auc = auc_val(tst_y, tst_scores[:,1])
	# if tehy are not only positive, then we train alphas and use them to compute 
	# new scores - auc vals on the partial validation and full test dataset
	else
		lr = LogReg()
		fit!(lr, scores, labels)
		val_probs = predict(lr, scores)
		tst_probs = predict(lr, tst_scores)
		val_prec = sum(labels)/length(labels)
		val_auc = auc_val(labels, val_probs)
		tst_auc = auc_val(tst_y, tst_probs)
	end
	return val_prec, val_auc, tst_auc
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

function compute_alpha_scores(model_id, lf) 
	# load the saved scores
	ldata = load(joinpath(latent_dir, lf))
	rf = filter(x->occursin("$(model_id)", x), rfs)
	if length(rf) < 1
		@info "Something is wrong, original score file for $lf not found"
		return
	end
	rf = rf[1]
	rdata = load(joinpath(res_dir, rf))

	# prepare the data
	tr_scores = cat(rdata[:tr_scores], transpose(ldata[:tr_scores]), dims=2);
	val_scores = cat(rdata[:val_scores], transpose(ldata[:val_scores]), dims=2);
	tst_scores = cat(rdata[:tst_scores], transpose(ldata[:tst_scores]), dims=2);
	tr_y = ldata[:tr_labels];
	val_y = ldata[:val_labels];
	tst_y = ldata[:tst_labels];

	# prepare the result dataframe
	res_df = OrderedDict()
	res_df["modelname"] = modelname
	res_df["dataset"] = dataset
	res_df["phash"] = GenerativeAD.Evaluation.hash(rdata[:parameters])
	res_df["parameters"] = "latent_score_type=$(latent_score_type)_"* split(rf, ".bson")[1]
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
	for p in [0.0001, 0.001, 0.01, 0.05, 0.1, 0.2]
		ip = p >= 0.01 ? 1 : 2
		sp = split("$(p*100)", ".")[ip]
		res_df["val_pat_$(sp)"], res_df["val_auc_$(sp)"], res_df["tst_auc_$(sp)"] = 
			perf_at_p_original(p, val_scores, val_y, tst_scores, tst_y)
	end

	# then save it
	res_df = DataFrame(res_df)
	outf = joinpath(save_dir, "model_id=$(model_id)_score=$(latent_score_type)_method=$(method).bson")
	save(outf, Dict(:df => res_df))
	@info "Saved $outf."
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
		save_dir = datadir("sgad_alpha_evaluation/images_$(datatype)/$(modelname)/$(dataset)/ac=$(ac)/seed=$(seed)")
		mkpath(save_dir)

		# top score files
		res_dir = datadir("experiments/images_$(datatype)/$(modelname)/$(dataset)/ac=$(ac)/seed=$(seed)")
		rfs = readdir(res_dir)
		rfs = filter(x->occursin(score_type, x), rfs)

		for (model_id, lf) in zip(model_ids, lfs)
			compute_alpha_scores(model_id, lf)
		end
	end
end
