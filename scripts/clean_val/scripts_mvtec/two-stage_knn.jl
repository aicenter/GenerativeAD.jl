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
		default = 1
		arg_type = Int
		help = "max_seed"
	"category"
		default = "wood"
		arg_type = String
		help = "dataset"
	"contamination"
		arg_type = Float64
		help = "contamination rate of training data"
		default = 0.0
end
parsed_args = parse_args(ARGS, s)
@unpack category, max_seed, contamination = parsed_args

#######################################################################################
################ THIS PART IS TO BE PROVIDED FOR EACH MODEL SEPARATELY ################

tab_name = "vae_LOSS_images_mvtec"
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
max_tries = 1
cont_string = (contamination == 0.0) ? "" : "_contamination-$contamination"
while try_counter < max_tries
	for seed in 1:max_seed
		savepath = datadir("experiments/images_mvtec_clean_val_default$cont_string/$(modelname)/$(category)/seed=$(seed)")
		aux_info = (model_index=mi, criterion=criterion)

		# load data
		global data = GenerativeAD.load_data("MVTec-AD", seed=seed, category=category, 
			contamination=contamination, img_size=64)
		not_loaded = true
		while not_loaded
			try
				global data, encoding_name, encoder_params = 
				GenerativeAD.Models.load_encoding(tab_name, data, 1, dataset=category, seed=seed, model_index=mi)
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
			training_info, results = fit(data, parameters, aux_info)
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

