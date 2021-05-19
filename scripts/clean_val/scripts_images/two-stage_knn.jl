using DrWatson
@quickactivate
using ArgParse
using GenerativeAD
import StatsBase: fit!, predict
using StatsBase
using BSON
# because of vae and 2stage
using DataFrames
using CSV
using ValueHistories
using Flux
using ConditionalDists
using GenerativeModels
import GenerativeModels: VAE
using Distributions
using DistributionsAD

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
	"anomaly_class"
		arg_type = Int
		default = 1
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
@unpack dataset, max_seed, anomaly_class, method, contamination = parsed_args

#######################################################################################
################ THIS PART IS TO BE PROVIDED FOR EACH MODEL SEPARATELY ################

tab_name = "vae_LOSS_images"
tab_name = (method == "leave-one-in") ? "$(tab_name)_leave-one-in" : tab_name
mi = 1
sp = split(tab_name, "_")
enc = sp[1]
criterion = lowercase(sp[2])

modelname = "$(enc)_knn"


# max(10, 0.03n)
function set_params(data)
	D, N = size(data[1][1])
	nn = max(10, ceil(Int,0.03*N))
	return (k=nn,)
end

function fit(data, parameters, aux_info)
	# construct model - constructor should only accept kwargs
	model = GenerativeAD.Models.knn_constructor(;v=:kappa, parameters...)

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
	parameters = merge(parameters, aux_info)
	training_info, [(x -> knn_predict(model, x, v), merge(parameters, (distance = v,))) for v in [:delta]]
end


####################################################################
################ THIS PART IS COMMON FOR ALL MODELS ################
# set a maximum for parameter sampling retries
try_counter = 0
max_tries = 10*max_seed
cont_string = (contamination == 0.0) ? "" : "_contamination-$contamination"
while try_counter < max_tries
	for seed in 1:max_seed
		i = anomaly_class
		savepath = datadir("experiments/images_$(method)_clean_val_default$cont_string/$(modelname)/$(dataset)/ac=$(i)/seed=$(seed)")
		aux_info = (model_index=mi, criterion=criterion)

		global data = GenerativeAD.load_data(dataset, seed=seed, anomaly_class_ind=i, method=method, contamination=contamination)
		not_loaded = true
		while not_loaded
			try
				global data, encoding_name, encoder_params = GenerativeAD.Models.load_encoding(tab_name, data, i, dataset=dataset, seed=seed, model_index=mi)
				not_loaded = false
			catch e		
				@info "model index $mi not working, trying the next one"
				global mi += 1
			end
		end
		parameters = set_params(data)

		# here, check if a model with the same parameters was already tested
		@info "Trying to fit $modelname on $dataset with parameters $(parameters)..."
		if GenerativeAD.check_params(savepath, merge(parameters, aux_info))
			training_info, results = fit(data, parameters, aux_info)
			# here define what additional info should be saved together with parameters, scores, labels and predict times
			save_entries = merge(training_info, (modelname = modelname, 
												 seed = seed, 
												 dataset = dataset, 
												 anomaly_class = i, 
												 encoder=encoding_name,
												 encoder_params=encoder_params,
												 model_index=mi,
												 criterion=criterion,
												 contamination=contamination))
			# now loop over all anomaly score funs
			for result in results
				GenerativeAD.experiment(result..., data, savepath; save_entries...)
			end
			global try_counter = max_tries + 1
		else
			@info "Model already present, sampling new hyperparameters..."
			global try_counter += 1
		end
	end
end
(try_counter == max_tries) ? (@info "Reached $(max_tries) tries, giving up.") : nothing

