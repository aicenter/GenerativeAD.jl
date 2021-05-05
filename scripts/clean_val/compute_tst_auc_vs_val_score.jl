using GenerativeAD
using FileIO, BSON
using ValueHistories, DistributionsAD
using Flux
using ConditionalDists
using GenerativeModels
using EvalMetrics
using Plots
using Statistics
using DrWatson

dataset = "arrhythmia"
dataset = "wall-following-robot"
dataset = "yeast"
dataset = "letter-recognition"
dataset = "kdd99_small"
seed=1
outpath = datadir("normal_validation/tst_auc_vs_val_score")
mkpath(outpath)
masterpath = datadir("experiments/tabular/")
models = [
	"aae_full", "adVAE", "GANomaly", "vae_full", "wae_full", "abod", "hbos", "if", "knn", "loda", "lof",
	"ocsvm_rbf", "ocsvm", "pidforest", "MAF", "RealNVP", "sptn", "fmgan", "gan", "MO_GAAL", "DeepSVDD", 
	"vae_knn", "vae_ocsvm"
	]

"""
Returns AUC and mean anomaly score of normal validation samples.
"""
function auc_val_score(f)
	data = load(f)
	try
		auc = EvalMetrics.auc_trapezoidal(EvalMetrics.roccurve(data[:tst_labels], vec(data[:tst_scores]))...)
		val_score = Flux.mean(data[:val_scores][data[:val_labels].==0])
		return auc, val_score
	catch e
		return NaN, NaN
	end
end

# loop over all models, create plots and compute correlation
function explore_scores(model, dataset, seed)
	sfs = readdir(joinpath(masterpath, model, dataset, "seed=$seed"), join=true)
	# filter out modelfiles
	sfs = filter(x->!(occursin("model", x)),sfs)
	# filter only sampled reconstruction scores
	if any(occursin.("score=reconstruction-sampled", sfs))
		sfs = filter(x->occursin("reconstruction-sampled", x),sfs)
	end
	res = map(auc_val_score, sfs)
	aucs = [x[1] for x in res]
	val_scores = [x[2] for x in res]
	valid_inds = .!isnan.(val_scores)
	aucs, val_scores = aucs[valid_inds], val_scores[valid_inds]
	cor = round(Statistics.cor(val_scores, aucs), digits=3)
	scatter(val_scores, aucs, xlabel="validation anomaly score - normal data", ylabel="test AUC", title="$model - $dataset, correlation=$(cor)")
	sf = joinpath(outpath, "$(dataset)_$(model)_val_score_tst_auc_cor=$(cor).png")
	savefig(sf)
	@info "Saved $sf."
	return aucs, val_scores
end

for model in models
	@info "Processing $model"
	try
		explore_scores(model, dataset, seed)
	catch e
		@warn "Model $model not processed"
	end
end