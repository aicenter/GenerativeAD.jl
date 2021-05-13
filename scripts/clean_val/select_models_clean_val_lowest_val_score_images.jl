using GenerativeAD
using FileIO, BSON
using ValueHistories, DistributionsAD
using Flux
using ConditionalDists
using GenerativeModels
using DrWatson
using StatsBase
using EvalMetrics
using LinearAlgebra

# all the models will use the lowest anomaly score on normal validation data
master_path = datadir("experiments/images_leave-one-out")
outpath = datadir("experiments/images_leave-one-out_clean_val_score")
mkpath(outpath)

models = readdir(master_path)
seeds = 1:1
acs = 1:10

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

function select_best_model(mp, dataset, model, acs)
	dp = joinpath(mp, dataset)
	sps = map(ac->joinpath(dp, "ac=$ac/seed=1"), acs) # paths of seed results
	sfs = map(x->filter(f->(f[1:5] != "model"), isdir(x) ? readdir(x) : []), sps) # all filenames
	scores = map(collect_scores, sps) # all scores
	
	best_models = []
	for (sf, score) in zip(sfs, scores)
		best_ind = argmin(score)
		push!(best_models, sf[best_ind])
	end
		
	if length(best_models) > 0
		return best_models, sps
	else
		return nothing, nothing
	end
end

function copy_models(acs, sps, best_models, dataset, outpath, model)
	for (ac, sp, bm) in zip(acs, sps, best_models)
		f = joinpath(sp, bm)
		if isfile(f)
			outp = joinpath(outpath, model, dataset, "ac=$ac/seed=1")
			mkpath(outp)
			outf = joinpath(outp, bm)
			cp(f, outf, force=true)
		end
	end
end

model = models[1]
dataset = "MNIST"

for model in models[1:3]
	@info "Processing model $model..."
	mp = joinpath(master_path, model)
	datasets = readdir(mp)
	for dataset in datasets
		@info "   $dataset"
		best_models, sps = select_best_model(mp, dataset, model, acs)
		if !isnothing(best_models)
			copy_models(acs, sps, best_models, dataset, outpath, model)
		else
			@warn "No usable data found, skipping"
		end
	end
end
