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
		default = 1
		arg_type = Int
		help = "max_seed"
	"category"
		default = "wood"
		arg_type = String
		help = "category"
	"contamination"
		arg_type = Float64
		help = "contamination rate of training data"
		default = 0.0
end
parsed_args = parse_args(ARGS, s)
@unpack max_seed, category, contamination = parsed_args

#######################################################################################
################ THIS PART IS TO BE PROVIDED FOR EACH MODEL SEPARATELY ################

tab_name = "vae_LOSS_images_mvtec"
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
max_tries = 1
cont_string = (contamination == 0.0) ? "" : "_contamination-$contamination"
while try_counter < max_tries	
	for seed in 1:max_seed
		savepath = datadir("experiments/images_mvtec_clean_val_default$cont_string/$(modelname)/$(category)/seed=$(seed)")
		mkpath(savepath)

		aux_info = (model_index=mi, criterion=criterion)

		global data = GenerativeAD.load_data("MVTec-AD", seed=seed, category=category, 
					contamination=contamination, img_size=128)
		not_loaded = true
		while not_loaded
			try
				global data, encoding_name, encoder_params = GenerativeAD.Models.load_encoding(tab_name, data, 1, dataset=category, seed=seed, model_index=mi)
				not_loaded = false
			catch e		
				println(e)
				@info "model index $mi not working, trying the next one"
				global mi += 1
			end
		end
		parameters = set_params(data)

		# here, check if a model with the same parameters was already tested
		@info "Trying to fit $modelname on $category with parameters $(parameters)..."
		if GenerativeAD.check_params(savepath, merge(parameters, aux_info))
			training_info, results = fit(data, parameters, 1, aux_info)
			# here define what additional info should be saved together with parameters, scores, labels and predict times
			save_entries = merge(training_info, (modelname = modelname, 
												 seed = seed, 
												 dataset = "MVTec-AD_$category",
												 category = category, 
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
