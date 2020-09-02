using ArgParse
using GenerativeAD
import StatsBase: fit!, predict
using StatsBase
using DrWatson
@quickactivate
using BSON

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

#######################################################################################
################ THIS PART IS TO BE PROVIDED FOR EACH MODEL SEPARATELY ################
modelname = "knn"
# sample parameters, should return a Dict of model kwargs 
function sample_params()
	par_vec = (1:2:101,)
	argnames = (:k,)
	return Dict(zip(argnames, map(x->sample(x, 1)[1], par_vec)))
end
# check if the model with given parameters wasn't already trained and saved
function check_params(savepath, parameters, data)
	# TODO
	return true
end
# this is the most important function - returns training info and a tuple or a vector of tuples (score_fun, final_parameters)
# info contains additional information on the training process, the same for all anomaly score functions
# each element of the return vector contains a specific anomaly score function - there can be multiple for each trained model
# final parameters is a dict of names and hyperparameter values that are used for creation of the savefile name
function fit(data, parameters)
	# construct model
	model = GenerativeAD.Models.knn_constructor(v=:kappa; parameters...)

	# fit train data
	try
		global info, fit_t, _, _, _ = @timed fit!(model, data[1][1])
	catch e
		return Dict(:fit_t => NaN), (nothing, nothing)
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

# get params, initialize the model, train it, predict scores and save everything
data = GenerativeAD.load_data(dataset, seed=seed)

try_counter = 0
max_tries = 100
while try_counter < max_tries 
	parameters = sample_params()
	# here, check if a model with the same parameters was already tested
	if check_params(savepath, parameters, data)
		# 
		training_info, results = fit(data, parameters)
		# here define what additional info should be saved together with parameters, scores, labels and predict times
		save_entries = Dict(
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
