using DrWatson
@quickactivate
using ArgParse
using GenerativeAD
using GenerativeAD.Models: anomaly_score, generalized_anomaly_score_gpu
using BSON
using PyCall
using StatsBase: fit!, predict, sample

using Flux
using MLDataPattern
using OrderedCollections

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

modelname ="GANomaly"

function sample_params()
	parameters_rng = (
		hdim   		= 2 .^(4:9), 
		zdim   		= 2 .^(1:8), 
		nlayers   	= 3:4, 
		activation  = ["relu", "swish", "tanh"],
		lr   		= 10f0 .^ (-4:-3), 
		decay   	= 0f0:0.1:0.5,
		batch_size  = 2 .^ (5:7), 
		w1 			= [1,10:10:90 ...],
		w2 			= [1,10:10:90 ...],
		w3 			= [1,10:10:90 ...],
		init_seed   = 1:Int(1e8),
	)

	return (;zip(keys(parameters_rng), map(x->sample(x, 1)[1], parameters_rng))...)
end

function create_space()
	pyReal = pyimport("skopt.space")["Real"]
	pyInt = pyimport("skopt.space")["Integer"]
	pyCat = pyimport("skopt.space")["Categorical"]
	
	parameters_rng = (
		hdim   		= pyInt(4, 9,						          name="log2_hdim"),
		zdim   		= pyInt(1, 8,						          name="log2_zdim"), 
		nlayers   	= pyInt(3, 4,						          name="nlayers"), 
		activation  = pyCat(categories=["relu", "swish", "tanh"], name="activation"),
		lr   		= pyReal(1f-5, 1f-3, prior="log-uniform",	  name="lr"), 
		#decay   	= pyReal(1f-5, 5f-1, prior="log-uniform", 	  name="decay"),
		batch_size  = pyInt(5, 7, 								  name="log2_batch_size"),
		w1			= pyInt(1, 90, 								  name="w1"),
		w2			= pyInt(1, 90, 								  name="w2"),
		w3			= pyInt(1, 90, 								  name="w3"),
	)
end



function fit(data, parameters)
	weights = [parameters[:w1],parameters[:w2],parameters[:w3]] # change :weights -> :w1, :w2, :w3 because of bayes optimization
	default_params = (iters=10000, check_every=30, patience=10, weights=weights,)

	all_parameters = merge(parameters, default_params)

	# define models (Generator, Discriminator)
	generator, discriminator = GenerativeAD.Models.tabular_ganomaly_constructor(all_parameters)

	# define optimiser
	try
		global info, fit_t, _, _, _ = @timed fit!(generator|>gpu, discriminator|>gpu, data, all_parameters)
	catch e
		println("Error caught.")
		return (fit_t = NaN, model = nothing, history = nothing, n_parameters = NaN), []
	end

	training_info = (
		fit_t = fit_t,
		model = (info[2]|>cpu, info[3]|>cpu),
		history = info[1], # losses through time
		npars = info[4], # number of parameters
		iters = info[5] # optim iterations of model
		)

	return training_info, [(x -> GenerativeAD.Models.anomaly_score(generator|>cpu, x; dims=1)[:], parameters)]
end

#_________________________________________________________________________________________________

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
								sample_params; add_model_seed=true)
	else
		parameters = sample_params()
	end
	
    for seed in 1:max_seed
		savepath = joinpath(dataset_folder, "seed=$(seed)")
		mkpath(savepath)

		# get data
		data = GenerativeAD.load_data(dataset, seed=seed, contamination=contamination)

		# update parameter
		parameters = merge(parameters, (idim=size(data[1][1],1), ))
		# here, check if a model with the same parameters was already tested
		if GenerativeAD.check_params(savepath, parameters)
			#(X_train,_), (X_val, y_val), (X_test, y_test) = data
			training_info, results = fit(data, parameters)
			# saving model separately
			if training_info.model !== nothing
				tagsave(joinpath(savepath, savename("model", parameters, "bson", digits=5)), Dict("model"=>training_info.model), safe = true)
				training_info = merge(training_info, (model = nothing,))
			end
			save_entries = merge(training_info, (modelname = modelname, seed = seed, dataset = dataset, contamination = contamination))
			# now loop over all anomaly score funs
			all_scores = [GenerativeAD.experiment(result..., data, savepath; save_entries...) for result in results]
			if sampling == "bayes" && length(all_scores) > 0
				@info("Updating cache with $(length(all_scores)) results.")
				GenerativeAD.update_bayes_cache(dataset_folder, 
						all_scores; ignore=Set([:init_seed, :L, :idim]))
			end
			global try_counter = max_tries + 1
		else
			@info "Model already present, sampling new hyperparameters..."
			global try_counter += 1
		end
	end
end
(try_counter == max_tries) ? (@info "Reached $(max_tries) tries, giving up.") : nothing