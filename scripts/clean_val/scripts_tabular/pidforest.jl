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
    "contamination"
    	arg_type = Float64
    	help = "contamination rate of training data"
    	default = 0.0
end
parsed_args = parse_args(ARGS, s)
@unpack dataset, max_seed, contamination = parsed_args

modelname = "pidforest"
function set_params(data)
	D, N = size(data[1][1])
	if N > 100
		ms = 100
	else
		ms = max(25, N - mod(N, 50))
	end
	return (max_depth=10, n_trees=50, max_samples=ms, max_buckets=3, epsilon=0.1)
end

function fit(data, parameters)
	model = GenerativeAD.Models.PIDForest(Dict(pairs(parameters)))

	try
		global info, fit_t, _, _, _ = @timed fit!(model, data[1][1])
	catch e
		@info "Failed training due to \n$e"
		return (fit_t = NaN,), []
	end

	training_info = (
		fit_t = fit_t,
		model = nothing
		)

	training_info, [(x -> predict(model, x, pct=p), merge(parameters, Dict(:percentile => p))) for p in [25]]
end

function remove_constant_features(data)
	X = data[1][1]
	mask = (maximum(X, dims=2) .== minimum(X, dims=2))[:]
	if any(mask)
		@info "Removing $(sum(mask)) features with constant values."
		return Tuple((data[i][1][.~mask,:], data[i][2]) for i in 1:3)
	end
	data
end

try_counter = 0
max_tries = 10*max_seed
cont_string = (contamination == 0.0) ? "" : "_contamination-$contamination"
while try_counter < max_tries
    for seed in 1:max_seed
		savepath = datadir("experiments/tabular_clean_val_default/$(modelname)/$(dataset)/seed=$(seed)")
		mkpath(savepath)

		# get data
		data = GenerativeAD.load_data(dataset, seed=seed, contamination=contamination)
		data = remove_constant_features(data)
		parameters = set_params(data)

		if GenerativeAD.check_params(savepath, parameters)
			@info "Started training PIDForest$(parameters) on $(dataset):$(seed)"
			
			training_info, results = fit(data, parameters)
			save_entries = merge(training_info, (modelname = modelname, seed = seed, dataset = dataset, contamination = contamination))

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
