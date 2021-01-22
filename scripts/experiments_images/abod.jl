using DrWatson
@quickactivate
using ArgParse
using GenerativeAD
import StatsBase: fit!, predict
using StatsBase
using BSON

s = ArgParseSettings()
@add_arg_table! s begin
	"max_seed"
		default = 1	
		arg_type = Int
		help = "seed"
	"dataset"
		default = "MNIST"
		arg_type = String
		help = "dataset"
	"anomaly_classes"
		arg_type = Int
		default = 10
		help = "number of anomaly classes"
	"method"
		arg_type = String
		default = "leave-one-out"
		help = "method for data creation -> \"leave-one-out\" or \"leave-one-in\" "
    "contamination"
    	arg_type = Float64
    	help = "contamination rate of training data"
    	default = 0.0
end
parsed_args = parse_args(ARGS, s)
@unpack dataset, max_seed, anomaly_classes, method, contamination = parsed_args

#######################################################################################
################ THIS PART IS TO BE PROVIDED FOR EACH MODEL SEPARATELY ################
modelname = "abod"
# sample parameters, should return a Dict of model kwargs 

function sample_params()
	par_vec = (1:100,["fast"])
	argnames = (:n_neighbors, :method)
	return (;zip(argnames, map(x->sample(x, 1)[1], par_vec))...)
end
function GenerativeAD.edit_params(data, parameters)
	D, N = size(data[1][1])
	if N < parameters.n_neighbors
		# if there are not enough samples, set the number of neighbors to N-1 as in the Python implementation
		@info "Not enough samples in training, changing n_neighbors value from $(parameters.n_neighbors) to $(N-1)."
		return merge(parameters, (;n_neighbors = N-1))
	else 
		return parameters
	end
end
function fit(data, parameters)
	# construct model - constructor should only accept kwargs
	model = GenerativeAD.Models.ABOD(;parameters...)

	# fit train data
	try
		global info, fit_t, _, _, _ = @timed fit!(model, data[1][1])
	catch e
		# return an empty array if fit fails so nothing is computed
		return (fit_t = NaN,), [] 
	end

	# construct return information - put e.g. the model structure here for generative models
	training_info = (
		fit_t = fit_t,
		model = nothing
		)

	# now return the different scoring functions
	training_info, [(x->predict(model, x), parameters)]
end

####################################################################
################ THIS PART IS COMMON FOR ALL MODELS ################
# set a maximum for parameter sampling retries
try_counter = 0
max_tries = 10*max_seed
cont_string = (contamination == 0.0) ? "" : "_contamination-$contamination"
while try_counter < max_tries
	parameters = sample_params()

	for seed in 1:max_seed
		for i in 1:anomaly_classes
			savepath = datadir("experiments/images_$(method)$cont_string/$(modelname)/$(dataset)/ac=$(i)/seed=$(seed)")
			mkpath(savepath)

			# get data
			data = GenerativeAD.load_data(dataset, seed=seed, method=method, anomaly_class_ind=i)
			data = GenerativeAD.Datasets.vectorize(data)

			# edit parameters
			edited_parameters = GenerativeAD.edit_params(data, parameters)

			@info "Trying to fit $modelname on $dataset with parameters $(edited_parameters)..."
			# check if a combination of parameters and seed alread exists
			if GenerativeAD.check_params(savepath, edited_parameters)
				# fit
				training_info, results = fit(data, edited_parameters)
				# here define what additional info should be saved together with parameters, scores, labels and predict times
				save_entries = merge(training_info, (modelname = modelname, seed = seed, dataset = dataset, anomaly_class = i,
					contamination=contamination))

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
end
(try_counter == max_tries) ? (@info "Reached $(max_tries) tries, giving up.") : nothing

