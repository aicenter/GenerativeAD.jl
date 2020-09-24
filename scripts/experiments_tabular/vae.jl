using DrWatson
@quickactivate
using ArgParse
using GenerativeAD
import StatsBase: fit!, predict
using StatsBase
using BSON
using Flux

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
# for testing purposes
if length(ARGS) == 0
	push!(ARGS, "1")
	push!(ARGS, "iris")
end
parsed_args = parse_args(ARGS, s)
@unpack dataset, max_seed = parsed_args

#######################################################################################
################ THIS PART IS TO BE PROVIDED FOR EACH MODEL SEPARATELY ################
modelname = "vae"
# sample parameters, should return a Dict of model kwargs 
"""
	sample_params()

Should return a named tuple that contains a sample of model parameters.
"""
function sample_params()
	par_vec = (2 .^(3:8), 2 .^(4:9), 10f0 .^(-4:-2), 2 .^ (5:7), ["relu", "swish", "tanh"], 3:4)
	argnames = (:zdim, :hdim, :lr, :batchsize, :activation, :nlayers)
	parameters = (;zip(argnames, map(x->sample(x, 1)[1], par_vec))...)
	# ensure that zdim < hdim
	while parameters.zdim >= parameters.hdim
		parameters = merge(parameters, (zdim = sample(par_vec[1])[1],))
	end
	return parameters
end
"""
	fit(data, parameters)

This is the most important function - returns `training_info` and a tuple or a vector of tuples `(score_fun, final_parameters)`.
`training_info` contains additional information on the training process that should be saved, the same for all anomaly score functions.
Each element of the return vector contains a specific anomaly score function - there can be multiple for each trained model.
Final parameters is a named tuple of names and parameter values that are used for creation of the savefile name.
"""
function fit(data, parameters)
	# construct model - constructor should only accept kwargs
	model = GenerativeAD.Models.vae_constructor(;idim=size(data[1][1],1), parameters...)

	# fit train data
	try
		global info, fit_t, _, _, _ = @timed fit!(model, data; max_train_time=82800, 
			patience=200, check_interval=10, parameters...)
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
	training_info, [
		(x -> GenerativeAD.Models.reconstruction_score(info.model, x), merge(parameters, (score = "reconstruction",))),
		(x -> GenerativeAD.Models.latent_score(info.model, x), merge(parameters, (score = "latent",))),
		]
end
function GenerativeAD.edit_params(data, parameters)
	idim = size(data[1][1],1)
	# put the largest possible zdim where zdim < idim, the model tends to converge poorly if the latent dim is larger than idim
	if parameters.zdim >= idim
		zdims = 2 .^(1:8)
		zdim_new = zdims[zdims .< idim][end]
		parameters = merge(parameters, (zdim=zdim_new,))
	end
	parameters
end

####################################################################
################ THIS PART IS COMMON FOR ALL MODELS ################
# only execute this if run directly - so it can be included in other files
if abspath(PROGRAM_FILE) == @__FILE__
	# set a maximum for parameter sampling retries
	try_counter = 0
	max_tries = 10*max_seed
	while try_counter < max_tries
	    parameters = sample_params()

	    for seed in 1:max_seed
			savepath = datadir("experiments/tabular/$(modelname)/$(dataset)/seed=$(seed)")
			mkpath(savepath)

			# get data
			data = GenerativeAD.load_data(dataset, seed=seed)
			
			# edit parameters
			edited_parameters = GenerativeAD.edit_params(data, parameters)
			
			@info "Trying to fit $modelname on $dataset with parameters $(edited_parameters)..."
			# check if a combination of parameters and seed alread exists
			if GenerativeAD.check_params(savepath, data, edited_parameters)
				# fit
				training_info, results = fit(data, edited_parameters)

				# save the model separately			
				if training_info.model != nothing
					tagsave(joinpath(savepath, savename("model", parameters, "bson")), Dict("model"=>training_info.model), safe = true)
					training_info = merge(training_info, (model = nothing,))
				end

				# here define what additional info should be saved together with parameters, scores, labels and predict times
				save_entries = merge(training_info, (modelname = modelname, seed = seed, dataset = dataset))

				# now loop over all anomaly score funs
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
end
