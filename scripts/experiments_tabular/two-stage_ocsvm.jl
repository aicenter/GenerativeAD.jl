using DrWatson
@quickactivate
using ArgParse
using GenerativeAD
using GenerativeAD.Models
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
	"tab_name"
		required = true
		arg_type = String
		help = "name of tab -> example: vae_LOSS_tabular, wae-vamp_AUC_tabular"
	"mi_only"
		arg_type = Int
		default = -1
		help = "index of model in range 1 to 10 or -1 for all models"
	"contamination"
    	arg_type = Float64
    	help = "contamination rate of training data"
    	default = 0.0
end
parsed_args = parse_args(ARGS, s)
@unpack dataset, max_seed, tab_name, mi_only, contamination = parsed_args

#######################################################################################
################ THIS PART IS TO BE PROVIDED FOR EACH MODEL SEPARATELY ################

sp = split(tab_name, "_")
enc = sp[1]
criterion = lowercase(sp[2])

modelname = "$(enc)_ocsvm"

function sample_params()
	par_vec = (round.([10^x for x in -4:0.1:2],digits=5),["poly", "rbf", "sigmoid"],[0.01,0.5,0.99])
	argnames = (:gamma,:kernel,:nu)
	return (;zip(argnames, map(x->sample(x, 1)[1], par_vec))...)
end


function fit(data, parameters, aux_info)
	# construct model - constructor should only accept kwargs
	model = GenerativeAD.Models.OCSVM(;parameters...)

	# sumbsample and fit train data
	tr_data = data[1][1]
	M,N = size(tr_data)
	tr_data = tr_data[:, sample(1:N, min(10000, N), replace=false)]
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
	training_info, [(x->predict(model, x), merge(parameters, aux_info))]
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
		savepath = datadir("experiments/tabular$cont_string/$(modelname)/$(dataset)/seed=$(seed)")
		mi_indexes = (mi_only == -1) ? [1:10...] : [mi_only]
		for mi = mi_indexes # iteration through encodings
			aux_info = (model_index=mi, criterion=criterion)
			# get data
			data = GenerativeAD.load_data(dataset, seed=seed, contamination=contamination)
			output, encode_t, _, _, _ = @timed GenerativeAD.Models.load_encoding(tab_name, data, dataset=dataset, seed=seed, model_index=mi)
			data, encoding_name, encoder_params, encoder_fit_t = output
				
			# edit parameters
			edited_parameters = GenerativeAD.edit_params(data, parameters)
			
			@info "Trying to fit $modelname on $dataset with parameters $(edited_parameters)..."
			# check if a combination of parameters and seed alread exists
			if GenerativeAD.check_params(savepath, merge(edited_parameters, aux_info))
				# fit
				training_info, results = fit(data, edited_parameters, aux_info)
				# here define what additional info should be saved together with parameters, scores, labels and predict times
				save_entries = merge(training_info, (modelname = modelname, 
													 seed = seed, 
													 dataset = dataset, 
													 encoder=encoding_name,
													 encoder_fit_t = encoder_fit_t,
													 encoder_params=encoder_params, 
													 encode_t = encode_t,
													 model_index=mi,
													 criterion=criterion,
													 contamination = contamination))

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

