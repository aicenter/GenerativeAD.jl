using ArgParse

s = ArgParseSettings()
@add_arg_table! s begin
    "dataset"
        required = true
        arg_type = String
        help = "dataset"
   "seed"
        required = true
        arg_type = Int
        help = "seed"
end
parsed_args = parse_args(ARGS, s)
@unpack dataset, seed = parsed_args

using GMAD
import GMAD.Models.KNNAnomaly
import StatsBase: fit!, predict
using StatsBase
using DrWatson
@quickactivate
using BSON

# paths
savepath = datadir("experiments/knn/$(dataset)/seed=$(seed)") 

# get params, initialize the model, train it, predict scores and save everything
data = GMAD.load_data(dataset, seed=seed)

# sample params
function sample_params()
	p1v = 1:2:101
	p2v = [:gamma, :delta, :kappa] 
	argnames = (:k, :v)
	return map(x->sample(x, 1)[1], (p1v, p2v)), Dict(), argnames
end

function _predict(args...; kwargs...)
	try 
		return predict(args...; kwargs...)
	catch e
		if isa(e, ArgumentError)
			return NaN # or nothing?
		else
			rethrow(e)
		end
	end
end

function experiment(data, args...; kwargs...)
	#TODO handle exceptions here or on the level of predict/fit functions ? 
	tr_data, val_data, tst_data = data
	model = KNNAnomaly(args...; kwargs...)

	# fit train
	info, fit_t, _, _, _ = @timed fit!(model, tr_data[1])

	# extract scores
	tr_scores, tr_eval_t, _, _, _ = @timed _predict(model, tr_data[1])
	val_scores, val_eval_t, _, _, _ = @timed _predict(model, val_data[1])
	tst_scores, tst_eval_t, _, _, _ = @timed _predict(model, tst_data[1])

	# now save the stuff
	result = Dict(
		:seed => seed,
		:dataset => dataset,
		:args => args, 
		:argnames => argnames,
		:kwargs => kwargs,
		:info =>  info,
		:fit_t => fit_t,
		:tr_scores => tr_scores,
		:tr_labels => tr_data[2], 
		:tr_eval_t => tr_eval_t,
		:val_scores => val_scores,
		:val_labels => val_data[2], 
		:val_eval_t => val_eval_t,
		:tst_scores => tst_scores,
		:tst_labels => tst_data[2], 
		:tst_eval_t => tst_eval_t
		)
end


maxiter = 100
iter = 0
while iter < maxiter
	# TODO could this potentially run forever?
	args, kwargs, argnames = sample_params()
	savef = joinpath(savepath, savename(merge(Dict(zip(argnames, args)), kwargs))*".bson")

	if !isfile(savef)
		global iter += 1
		println("computing $iter/$maxiter")
		result = experiment(data, args...; kwargs...)
		tagsave(savef, result, safe = true)
		
	else
		println("model already present, not computing anything")
	end
end