using DrWatson
@quickactivate
using ArgParse
using PyCall
using GenerativeAD
using StatsBase: fit!, predict, sample
using OrderedCollections
using BSON

s = ArgParseSettings()
@add_arg_table! s begin
	"max_seed"
		default = 1
		arg_type = Int
		help = "maximum number of seeds to run through"
	"dataset"
		default = "iris"
		arg_type = String
		help = "dataset"
	"sampling"
		default = "random"
		arg_type = String
		help = "sampling of hyperparameters"
	"contamination"
		arg_type = Float64
		help = "contamination rate of training data"
		default = 0.0
end
parsed_args = parse_args(ARGS, s)
@unpack dataset, max_seed, sampling, contamination = parsed_args

modelname = "pidforest"
function sample_params()
	parameters_rng = (;
		max_depth	= 6:2:10, 
		n_trees		= 50:25:200, 
		max_samples	= [50, 100, 250, 500, 1000, 5000], 
		max_buckets	= 3:6, 
		epsilon		= [0.05, 0.1, 0.2]
	)

	return (;zip(keys(parameters_rng), map(x->sample(x, 1)[1], parameters_rng))...)
end

function create_space()
	pyReal = pyimport("skopt.space")["Real"]
	pyInt = pyimport("skopt.space")["Integer"]
	pyCat = pyimport("skopt.space")["Categorical"]
	
	(;
		max_depth	= pyInt(6, 10, 							name="max_depth"),
		n_trees		= pyInt(50, 200,						name="n_trees"),
		max_samples	= pyInt(25, 5000, prior="log-uniform",	name="max_samples"),
		max_buckets	= pyInt(3, 6, 							name="max_buckets"), 
		epsilon		= pyReal(0.05, 0.2, 					name="epsilon")
	)
end

function GenerativeAD.edit_params(data, parameters)
	D, N = size(data[1][1])
	if N < parameters.max_samples
		# if there are not enough samples, choose closest multiple of 50
		@info "Not enough samples in training, changing max_samples for each tree."
		return merge(parameters, (;max_samples = max(25, N - mod(N, 50))))
	else 
		return parameters
	end
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

	training_info, [(x -> predict(model, x, pct=p), merge(parameters, Dict(:percentile => p))) for p in [10, 25, 50]]
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
sampling_string = sampling == "bayes" ? "_bayes" : ""
prefix = "experiments$(sampling_string)/tabular$(cont_string)"
dataset_folder = datadir("$(prefix)/$(modelname)/$(dataset)")
while try_counter < max_tries
	if sampling == "bayes"
		parameters = GenerativeAD.bayes_params(
								create_space(), 
								dataset_folder,
								sample_params)
	else
		parameters = sample_params()
	end

	for seed in 1:max_seed
		savepath = joinpath(dataset_folder, "seed=$(seed)")
		mkpath(savepath)

		# get data
		data = GenerativeAD.load_data(dataset, seed=seed, contamination=contamination)
		data = remove_constant_features(data)
		edited_parameters = sampling == "bayes" ? parameters : GenerativeAD.edit_params(data, parameters)

		if GenerativeAD.check_params(savepath, edited_parameters)
			@info "Started training PIDForest$(edited_parameters) on $(dataset):$(seed)"
			
			training_info, results = fit(data, edited_parameters)
			save_entries = merge(training_info, (modelname = modelname, seed = seed, dataset = dataset, contamination = contamination))

			all_scores = [GenerativeAD.experiment(result..., data, savepath; save_entries...) for result in results]
			if sampling == "bayes" && length(all_scores) > 0
				@info("Updating cache with $(length(all_scores)) results.")
				GenerativeAD.update_bayes_cache(dataset_folder, 
						all_scores; ignore=Set([:percentile]))
			end
			global try_counter = max_tries + 1
		else
			@info "Model already present, trying new hyperparameters..."
			global try_counter += 1
		end
	end
end
(try_counter == max_tries) ? (@info "Reached $(max_tries) tries, giving up.") : nothing
