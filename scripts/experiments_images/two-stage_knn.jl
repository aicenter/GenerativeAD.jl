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
	"tab_name"
		required = true
		arg_type = String
		help = "name of tab -> example: vae_LOSS_images, wae-vamp_AUC_images"
	"anomaly_classes"
		arg_type = Int
		default = 10
		help = "number of anomaly classes"
	"mi_only"
		arg_type = Int
		default = -1
		help = "index of model in range 1 to 10 or -1 for all models"

	"method"
		arg_type = String
		default = "leave-one-out"
		help = "method for data creation -> \"leave-one-out\" or \"leave-one-in\" "

end
parsed_args = parse_args(ARGS, s)
@unpack dataset, max_seed, tab_name, anomaly_classes, mi_only, method = parsed_args

#######################################################################################
################ THIS PART IS TO BE PROVIDED FOR EACH MODEL SEPARATELY ################

sp = split(tab_name, "_")
enc = sp[1]
criterion = lowercase(sp[2])

modelname = "$(enc)_knn"

DrWatson.projectdir() = "/home/skvarvit/generativead/GenerativeAD.jl"

function sample_params()
	par_vec = (1:2:101,)
	argnames = (:k,)
	return (;zip(argnames, map(x->sample(x, 1)[1], par_vec))...)
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
	training_info, [(x -> knn_predict(model, x, v), merge(parameters, (distance = v,))) for v in [:gamma, :kappa, :delta]]
end


####################################################################
################ THIS PART IS COMMON FOR ALL MODELS ################
# set a maximum for parameter sampling retries
try_counter = 0
max_tries = 10*max_seed
while try_counter < max_tries
	parameters = sample_params()

	for seed in 1:max_seed
		for i in 1:anomaly_classes
			savepath = datadir("experiments/images_$(method)/$(modelname)/$(dataset)/ac=$(i)/seed=$(seed)")
			mi_indexes = (mi_only == -1) ? [1:10...] : [mi_only] 
			for mi = mi_indexes
				aux_info = (model_index=mi, criterion=criterion)

				data = GenerativeAD.load_data(dataset, seed=seed, anomaly_class_ind=i, method=method)
				data, encoding_name, encoder_params = GenerativeAD.Models.load_encoding(tab_name, data, i, dataset=dataset, seed=seed, model_index=mi)

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
														 criterion=criterion))
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
	end
end
(try_counter == max_tries) ? (@info "Reached $(max_tries) tries, giving up.") : nothing

