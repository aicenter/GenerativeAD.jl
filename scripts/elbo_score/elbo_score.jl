using DrWatson
@quickactivate
using ArgParse
using GenerativeAD
import StatsBase: fit!, predict
using StatsBase
using BSON, FileIO
using Flux
using GenerativeModels
using DistributionsAD
using ValueHistories

s = ArgParseSettings()
@add_arg_table! s begin
	"modelname"
		default = "vae"
		arg_type = String
		help = "model name"
	"datatype"
		default = "tabular"
		arg_type = String
		help = "tabular or image"
	"dataset"
		default = "iris"
		arg_type = String
		help = "dataset"
	"--seed"
		default = nothing
		help = "if specified, only results for a given seed will be recomputed"
	"--anomaly_class"
		default = nothing
		help = "if specified, only results for a given anomaly class will be recomputed"
end
parsed_args = parse_args(ARGS, s)
@unpack dataset, datatype, modelname, seed, anomaly_class = parsed_args

masterpath = datadir("experiments/$(datatype)/$(modelname)/$(dataset)")
files = GenerativeAD.Evaluation.collect_files(masterpath)
mfiles = filter(f->occursin("model", f), files)
if seed != nothing
	filter!(x->occursin("/seed=$seed/", x), mfiles)
end
if anomaly_class != nothing
	filter!(x->occursin("/ac=$(anomaly_class)/", x), mfiles)
end

kld_batched(m,x,batchsize) = 
	vcat(map(y->vec(Base.invokelatest(GenerativeModels.kl_divergence, 
			condition(m.encoder,y), m.prior)), Flux.Data.DataLoader(x, batchsize=batchsize))...)
kld_batched_gpu(m,x,batchsize) = 
	vcat(map(y-> cpu(vec(Base.invokelatest(GenerativeModels.kl_divergence, 
			condition(m.encoder,gpu(Array(y))), m.prior))), Flux.Data.DataLoader(x, batchsize=batchsize))...)

function save_elbo_score(f::String, data, seed::Int, ac=nothing)
	# get model
	savepath = dirname(f)
	mdata = load(f)
	model = mdata["model"]

	# setup entries to be saved
	save_entries = (
		modelname = modelname,
		fit_t = mdata["fit_t"],
		history = mdata["history"],
		dataset = dataset,
		npars = sum(map(p->length(p), Flux.params(model))),
		model = nothing,
		seed = seed
		)
	save_entries = (ac == nothing) ? save_entries : merge(save_entries, (anomaly_class=ac,))

	# test if the file alread exists
	parameters = merge(mdata["parameters"], (L=100, score = "elbo",))
	savef = joinpath(savepath, savename(parameters, "bson", digits=5))
	if !isfile(savef)
		@info "computing sample score for $f"

		# now load the sampled reconstruction probability score
		namepars = DrWatson.parse_savename(f)[2]
		sf = filter(x->
			occursin("init_seed=$(namepars["init_seed"])", x) && 
			occursin("score=reconstruction-sampled", x), 
			readdir(savepath))
		if length(sf)==0
			@info "sampled reconstruction score for $f not found!"
			return
		end
		sf = joinpath(savepath, sf[1])
		score_data = load(sf)
		rec_scores = (score_data[:tr_scores], score_data[:val_scores], score_data[:tst_scores])

		# compute the full scores
		score_fun(x) = (ac == nothing) ? 
			kld_batched(model, x, 512) : kld_batched_gpu(gpu(model), x, 512)
		tr_data, val_data, tst_data = data

		# extract scores
		tr_scores, tr_eval_t, _, _, _ = @timed score_fun(tr_data[1])
		val_scores, val_eval_t, _, _, _ = @timed score_fun(val_data[1])
		tst_scores, tst_eval_t, _, _, _ = @timed score_fun(tst_data[1])

		result = (
			parameters = parameters,
			tr_scores = tr_scores .+ rec_scores[1],
			tr_labels = tr_data[2], 
			tr_eval_t = tr_eval_t + score_data[:tr_eval_t],
			val_scores = val_scores .+ rec_scores[2],
			val_labels = val_data[2], 
			val_eval_t = val_eval_t + score_data[:val_eval_t],
			tst_scores = tst_scores .+ rec_scores[3],
			tst_labels = tst_data[2], 
			tst_eval_t = tst_eval_t + score_data[:tst_eval_t]
			)
		
		result = Dict{Symbol, Any}([sym=>val for (sym,val) in pairs(merge(result, save_entries))]) # this has to be a Dict 
		tagsave(savef, result, safe = true)
		(@info "Results saved to $savef")
	end
end

for f in mfiles
	# get data
	savepath = dirname(f)
	local seed = parse(Int, replace(basename(savepath), "seed=" => ""))
	ac = occursin("ac=", savepath) ? parse(Int, replace(basename(dirname(savepath)), "ac=" => "")) : nothing
	data = (ac == nothing) ? 
		GenerativeAD.load_data(dataset, seed=seed) : 
		GenerativeAD.load_data(dataset, seed=seed, anomaly_class_ind=ac)
		
	# compute and save the score
	save_elbo_score(f, data, seed, ac)
end
@info "DONE"
