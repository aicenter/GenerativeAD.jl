using ArgParse
using GenerativeAD
import StatsBase: fit!, predict
using StatsBase
using DrWatson
@quickactivate
using BSON

s = ArgParseSettings()
@add_arg_table! s begin
   "seed"
        required = true
        arg_type = Int
        help = "seed"
    "dataset"
        required = true
        arg_type = String
        help = "dataset"
end
parsed_args = parse_args(ARGS, s)
@unpack dataset, seed = parsed_args

#######################################################################################
################ THIS PART IS TO BE PROVIDED FOR EACH MODEL SEPARATELY ################
modelname = "knn"
# sample parameters, should return a Dict of model kwargs 
"""
	sample_params()

Should return a Dict that contains a sample of model parameters.
"""
function sample_params()
	par_vec = (1:2:101,)
	argnames = (:k,)
	return Dict(zip(argnames, map(x->sample(x, 1)[1], par_vec)))
end
"""
	edit_params(data, parameters)

This modifies parameters according to data - only useful for some models.
"""
function edit_params(data, parameters)
	eparams = copy(parameters)
	return eparams
end
"""
	fit(data, parameters)

This is the most important function - returns `training_info` and a tuple or a vector of tuples `(score_fun, final_parameters)`.
`training_info` contains additional information on the training process, the same for all anomaly score functions.
Each element of the return vector contains a specific anomaly score function - there can be multiple for each trained model.
Final parameters is a Dict of names and parameter values that are used for creation of the savefile name.
"""
function fit(data, parameters)
	# edit params if needed
	parameters = edit_params(data, parameters)

	# construct model - constructor should only accept kwargs
	model = GenerativeAD.Models.knn_constructor(;v=:kappa, parameters...)

	# fit train data
	try
		global info, fit_t, _, _, _ = @timed fit!(model, data[1][1])
	catch e
		# return an empty array if fit fails so nothing is computed
		return Dict(:fit_t => NaN), [] 
	end

	# construct return information 
	training_info = Dict(
		:fit_t => fit_t
		)

	# now return the different scoring functions
	function knn_predict(model, x, v::Symbol)
		try 
			return predict(model, x, v)
		catch e
			if isa(e, ArgumentError) # this happens in the case when k > number of points
				return NaN # or nothing?
			else
				rethrow(e)
			end
		end
	end
	training_info, [(x -> knn_predict(model, x, v), merge(parameters, Dict(:distance => v))) for v in [:gamma, :kappa, :delta]]
end

####################################################################
################ THIS PART IS COMMON FOR ALL MODELS ################
# paths
savepath = datadir("experiments/tabular/$(modelname)/$(dataset)/seed=$(seed)") 
mkpath(savepath)

# get params, initialize the model, train it, predict scores and save everything
data = GenerativeAD.load_data(dataset, seed=seed)

# set a maximum for parameter sampling retries
try_counter = 0
max_tries = 10
while try_counter < max_tries 
	parameters = sample_params()
	# here, check if a model with the same parameters was already tested
	if GenerativeAD.check_params(edit_params, savepath, data, parameters)
		# fit
		training_info, results = fit(data, parameters)
		# here define what additional info should be saved together with parameters, scores, labels and predict times
		save_entries = Dict(
				:model => modelname,
				:seed => seed,
				:dataset => dataset,
				:fit_t => training_info[:fit_t]
				)
		
		# now loop over all anomaly score funs
		for result in results
			GenerativeAD.experiment(result..., data, savepath; save_entries...)
		end
		break
	else
		@info "Model already present, sampling new hyperparameters..."
		global try_counter += 1
	end 
end
(try_counter == max_tries) ? (@info "Reached $(max_tries) tries, giving up.") : nothing
