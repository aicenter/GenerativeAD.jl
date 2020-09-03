using ArgParse
using GenerativeAD
import StatsBase: fit!, predict, sample
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

modelname = "pidforest"
function sample_params()
	par_vec = (6:2:10, 50:25:200, 50:50:200, 3:6, [0.05, 0.1, 0.2], [1], [0], )
	argnames = (:max_depth, :n_trees, :max_samples, :max_buckets, :epsilon, :sample_axis, :threshold,)

	return Dict(zip(argnames, map(x->sample(x, 1)[1], par_vec)))
end

function fit(data, parameters)
	model = GenerativeAD.Models.PIDForest(parameters)

	try
		global info, fit_t, _, _, _ = @timed fit!(model, data[1][1])
	catch e
		return Dict(:fit_t => NaN), (nothing, nothing)
	end

	training_info = Dict(
		:fit_t => fit_t
		)

	# there are parameters for the predict function, which could be specified here and put into parameters
	training_info, [(x -> predict(model, x), parameters)]
end

## test code 
# parameters = sample_params()
# dataset, seed = "statlog-satimage", 1
# data = GenerativeAD.load_data(dataset, seed=seed)
# model = PIDForest(parameters)
# @timed fit!(model, data[1][1])
# predict(model, data[3][1])
# training_info, results = fit(data, parameters)

savepath = datadir("experiments/tabular/$(modelname)/$(dataset)/seed=$(seed)") 

data = GenerativeAD.load_data(dataset, seed=seed)

try_counter = 0
max_tries = 2
while try_counter < max_tries 
	parameters = sample_params()
	# here, check if a model with the same parameters was already tested
	if check_params(savepath, parameters, data)
		training_info, results = fit(data, parameters)
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
