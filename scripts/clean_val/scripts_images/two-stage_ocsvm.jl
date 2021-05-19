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
using Distances
using Statistics

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
		help = "anomaly classes"
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
@unpack dataset, anomaly_class, max_seed, method, contamination = parsed_args

#######################################################################################
################ THIS PART IS TO BE PROVIDED FOR EACH MODEL SEPARATELY ################

tab_name = "vae_LOSS_images"
tab_name = (method == "leave-one-in") ? "$(tab_name)_leave-one-in" : tab_name
mi = 1
sp = split(tab_name, "_")
enc = sp[1]
criterion = lowercase(sp[2])
modelname = "$(enc)_ocsvm"

function median_l2_dist(X)
	dists = pairwise(Euclidean(), X)
	# take only the upper diagonal
	ds = []
	for i in 1:size(dists,1)
		for j in (i+1):size(dists,1)
			push!(ds, dists[i,j])
		end
	end
	ml=median(ds)
end
function set_params(data)
	D, N = size(data[1][1])
	gamma = 1/median_l2_dist(data[1][1])
	return (gamma=gamma, kernel="rbf", nu=0.5)
end

function joint_fit(models, data_splits)
	info = []
	for (model, data) in zip(models, data_splits)
		push!(info, fit!(model, data))
	end
	return info
end

function joint_prediction(models, data)
	joint_pred = Array{Float32}(undef, length(models), size(data,2))
	for (i,model) in enumerate(models)
		joint_pred[i,:] = predict(model, data)
	end
	return vec(mean(joint_pred, dims=1))
end

function StatsBase.fit(data, parameters, n_models, aux_info)
	# construct model - constructor should only accept kwargs
	models = [GenerativeAD.Models.OCSVM(;parameters...) for _ = 1:n_models]
	
	# sumbsample and fit train data
	tr_data = data[1][1]
	M,N = size(tr_data)
	split_size = floor(N/n_models)
	data_splits = [tr_data[:,Int(split_size*i+1):Int(split_size*(i+1))] for i = 0:n_models-1]

	try
		global info, fit_t, _, _, _ = @timed joint_fit(models, data_splits)
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
	training_info, [(x->joint_prediction(models, x),  merge(parameters, aux_info))]
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
			training_info, results = fit(data, parameters, 10, aux_info)
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
