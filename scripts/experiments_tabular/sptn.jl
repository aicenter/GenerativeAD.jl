using DrWatson
@quickactivate
using ArgParse
using BSON
using Flux
using PyCall
using GenerativeAD
using StatsBase: fit!, predict, sample
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

modelname = "sptn"

function sample_params()
	parameter_rng = (
		firstdense = [true, false], 
		batchsize = 2 .^ (5:7), 
		ncomp = 2 .^ (1:4), 
		nlayers = 1:3, 
		activation = [identity], 
		unitary = [:butterfly], 
		sharing = [:dense, :all, :none], 
		init_seed = 1:Int(1e8), 
	)	

	return (;zip(keys(parameter_rng), map(x->sample(x, 1)[1], parameter_rng))...)
end

function create_space()
	pyReal = pyimport("skopt.space")["Real"]
	pyInt = pyimport("skopt.space")["Integer"]
	pyCat = pyimport("skopt.space")["Categorical"]
	
	(;
		firstdense  = pyCat(categories=[true, false], 				name="firstdense"), 
		batchsize 	= pyInt(5, 7, 									name="log2_batchsize"),
		ncomp 		= pyInt(1, 4, 									name="log2_ncomp"),
		nlayers		= pyInt(1, 3, 									name="nlayers"),
		activation 	= pyCat(categories=["identity"], 				name="fun_activation"),
		unitary 	= pyCat(categories=["butterfly"], 				name="sym_unitary"),
		sharing 	= pyCat(categories=["dense", "all", "none"], 	name="sym_sharing")
	)
end

function fit(data, parameters)
	model = GenerativeAD.Models.SPTN(;idim=size(data[1][1],1), parameters...)

	try
		global info, fit_t, _, _, _ = @timed fit!(model, data; max_train_time=82800/max_seed, 
			patience=20, check_interval=10, parameters...)
	catch e
		# return an empty array if fit fails so nothing is computed
		@info "Failed training due to \n$e"
		return (fit_t = NaN, history=nothing, npars=nothing, model=nothing), [] 
	end

	# construct return information - put e.g. the model structure here for generative models
	training_info = (
		fit_t = fit_t,
		history = info.history,
		npars = info.npars,
		model = info.model
		)

	# now return the different scoring functions
	training_info, [(x -> predict(info.model, x), parameters)]
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
								sample_params; add_model_seed=true)
	else
		parameters = sample_params()
	end

	for seed in 1:max_seed
		savepath = joinpath(dataset_folder, "seed=$(seed)")
		mkpath(savepath)

		# get data
		data = GenerativeAD.load_data(dataset, seed=seed, contamination=contamination)
		edited_parameters = sampling == "bayes" ? parameters : GenerativeAD.edit_params(data, parameters)

		# check if a combination of parameters and seed alread exists
		if GenerativeAD.check_params(savepath, edited_parameters)
			@info "Started training $(modelname)$(edited_parameters) on $(dataset):$(seed)"
			@info "Train/valdiation/test splits: $(size(data[1][1], 2)) | $(size(data[2][1], 2)) | $(size(data[2][1], 2))"
			@info "Number of features: $(size(data[1][1], 1))"
			
			training_info, results = fit(data, edited_parameters)

			if training_info.model !== nothing
				tagsave(joinpath(savepath, savename("model", edited_parameters, "bson", digits=5)), 
						Dict("model"=>training_info.model,
							"fit_t"=>training_info.fit_t,
							"history"=>training_info.history,
							"parameters"=>edited_parameters
							), safe = true)
				training_info = merge(training_info, (model = nothing,))
			end
			# here define what additional info should be saved together with parameters, scores, labels and predict times
			save_entries = merge(training_info, (modelname = modelname, seed = seed, dataset = dataset, contamination = contamination))

			# now loop over all anomaly score funs
			all_scores = [GenerativeAD.experiment(result..., data, savepath; save_entries...) for result in results]
			if sampling == "bayes" && length(all_scores) > 0
				@info("Updating cache with $(length(all_scores)) results.")
				GenerativeAD.update_bayes_cache(dataset_folder, 
						all_scores; ignore=Set([:init_seed]))
			end
			global try_counter = max_tries + 1
		else
			@info "Model already present, trying new hyperparameters..."
			global try_counter += 1
		end
	end
end
(try_counter == max_tries) ? (@info "Reached $(max_tries) tries, giving up.") : nothing
