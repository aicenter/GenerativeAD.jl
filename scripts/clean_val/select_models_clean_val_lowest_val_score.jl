using GenerativeAD
using FileIO, BSON
using ValueHistories, DistributionsAD
using Flux
using ConditionalDists
using GenerativeModels
using DrWatson
using StatsBase
using EvalMetrics

# all the models will use the lowest anomaly score on normal validation data
master_path = datadir("experiments/tabular")
outpath = datadir("experiments/tabular_clean_val_score")
mkpath(outpath)
models = [
	"aae_full", "adVAE", "GANomaly", "vae_full", "wae_full", "abod", "hbos", "if", "knn", "loda", "lof",
	"ocsvm_rbf", "ocsvm", "pidforest", "MAF", "RealNVP", "sptn", "fmgan", "gan", "MO_GAAL", "DeepSVDD", 
	"vae_knn", "vae_ocsvm"
	]
seeds = 1:5

# do vae_knn and vae_ocsvm

function val_score(f)
	try
		data = load(f)
		val_score = Flux.mean(data[:val_scores][data[:val_labels].==0])
		return val_score
	catch e
		return NaN
	end
end
function collect_scores(path)
	try
		fs = readdir(path)
		fs = filter(f->(f[1:5] != "model"), fs)
		scores = map(f->val_score(joinpath(path,f)), fs)
		return scores
	catch e
		@warn "$path does not probably exist"
		return []
	end
end


function test_auc(f)
	try
		data = load(f)
		auc = EvalMetrics.auc_trapezoidal(EvalMetrics.roccurve(data[:tst_labels], vec(data[:tst_scores]))...)
		return auc
	catch e
		return NaN
	end
end
function collect_aucs(path)
	fs = readdir(path)
	fs = filter(x->!(occursin("model", x)), fs)
	scores = map(f->test_auc(joinpath(path,f)), fs)
end

function select_best_model(mp, dataset, model, seeds)
	dp = joinpath(mp, dataset)
	sps = map(s->joinpath(dp, "seed=$s"), seeds) # paths of seed results
	sfs = map(x->filter(f->(f[1:5] != "model"), isdir(x) ? readdir(x) : []), sps) # all filenames
	scores = map(collect_scores, sps) # all scores
	ufs = unique(vcat(sfs...)) # unique filenames
	
	potential_models = [] # models that have at least 3 seeds
	potential_model_scores = [] # their mean scores across all seeds
	for uf in ufs
		mscores = []
		for s in seeds
			if length(sfs[s]) == 0
				continue
			end
			score_inds = sfs[s] .== uf
			if sum(score_inds) > 0 # in case some model is not available for all seeds
				push!(mscores, scores[s][score_inds][1])
			end
		end
		valid_inds = .!isnan.(mscores)
		if sum(valid_inds)>=3
			final_val = mean(mscores[valid_inds])
			if !isnan(final_val)
				push!(potential_models, uf)
				push!(potential_model_scores, final_val)
			end
		end
	end
	if length(potential_model_scores) > 0
		best_ind = argmin(potential_model_scores)
		best_model = potential_models[best_ind]
		return best_model, sps
	else
		return nothing, nothing
	end
end
function copy_models(seeds, sps, best_model, dataset, outpath, model)
	for (seed, sp) in zip(seeds, sps)
		f = joinpath(sp, best_model)
		if isfile(f)
			outp = joinpath(outpath, model, dataset, "seed=$seed")
			mkpath(outp)
			outf = joinpath(outp, best_model)
			cp(f, outf, force=true)
		end
	end
end

model = "vae_knn"
dataset = "iris"

for model in models
	@info "Processing model $model..."
	mp = joinpath(master_path, model)
	datasets = readdir(mp)
	for dataset in datasets
		@info "   $dataset"
		best_model, sps = select_best_model(mp, dataset, model, seeds)
		if !isnothing(best_model)
			copy_models(seeds, sps, best_model, dataset, outpath, model)
		else
			@warn "No usable data found, skipping"
		end
	end
end
