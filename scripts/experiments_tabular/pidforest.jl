using DrWatson
@quickactivate
using ArgParse
using GenerativeAD
using StatsBase: fit!, predict, sample
using BSON

s = ArgParseSettings()
@add_arg_table! s begin
   "max_seed"
        required = true
        arg_type = Int
        help = "seed"
    "dataset"
        required = true
        arg_type = String
        help = "dataset"
end
parsed_args = parse_args(ARGS, s)
@unpack dataset, max_seed = parsed_args

modelname = "pidforest"
function sample_params()
	par_vec = (6:2:10, 50:25:200, 50:50:200, 3:6, [0.05, 0.1, 0.2], )
	argnames = (:max_depth, :n_trees, :max_samples, :max_buckets, :epsilon, )

	return (;zip(argnames, map(x->sample(x, 1)[1], par_vec))...)
end

function fit(data, parameters)
	model = GenerativeAD.Models.PIDForest(Dict(pairs(parameters)))

	try
		global info, fit_t, _, _, _ = @timed fit!(model, data[1][1])
	catch e
		@info "Failed training PIDForest$(parameters) due to \n$e"
		return (fit_t = NaN,), []
	end

	training_info = (
		fit_t = fit_t,
		model = nothing
		)

	training_info, [(x -> predict(model, x, pct=p), merge(parameters, Dict(:percentile => p))) for p in [10, 25, 50]]
end

try_counter = 0
max_tries = 10
while try_counter < max_tries
    parameters = sample_params()

    for seed in 1:max_seed
		savepath = datadir("experiments/tabular/$(modelname)/$(dataset)/seed=$(seed)")
		mkpath(savepath)

		data = GenerativeAD.load_data(dataset, seed=seed)
		edited_parameters = GenerativeAD.edit_params(data, parameters)

		if GenerativeAD.check_params(savepath, data, edited_parameters)
			@info "Started training on $(dataset):$(seed)"
			
			training_info, results = fit(data, edited_parameters)
			save_entries = merge(training_info, (modelname = modelname, seed = seed, dataset = dataset))

			for result in results
				GenerativeAD.experiment(result..., data, savepath; save_entries...)
			end
			global try_counter = max_tries + 1
		else
			@info "Model already present, trying new hyperparameters..."
			global try_counter += 1
		end
	end
end
(try_counter == max_tries) ? (@info "Reached $(max_tries) tries, giving up.") : nothing
