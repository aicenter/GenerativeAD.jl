using DrWatson
@quickactivate
using ArgParse
using GenerativeAD
import StatsBase: fit!, predict
using StatsBase
using BSON
using Flux
using GenerativeModels
using Random
using HierarchicalAD
using HierarchicalAD: HAD, fvlae_constructor, knn_constructor, predict

s = ArgParseSettings()
@add_arg_table! s begin
   "max_seed"
        arg_type = Int
        help = "seed"
        default = 1
    "dataset"
        default = "iris"
        arg_type = String
        help = "dataset"
    "contamination"
    	arg_type = Float64
    	help = "contamination rate of training data"
    	default = 0.0
end
parsed_args = parse_args(ARGS, s)
@unpack dataset, max_seed, contamination = parsed_args

#######################################################################################
################ THIS PART IS TO BE PROVIDED FOR EACH MODEL SEPARATELY ################
modelname = "had_basic"
# sample parameters, should return a Dict of model kwargs 

function linear_hdims(zdim, n_latents)
	hdim_bounds = 2 .^(4:9)
	hdim_min = sample(hdim_bounds[hdim_bounds .> zdim])
    hdim_max = sample(hdim_bounds[hdim_bounds .>= hdim_min])
	hdims = floor.(Int, collect(range(hdim_max, hdim_min, length=n_latents)))
end
"""
	sample_params()

Should return a named tuple that contains a sample of model parameters.
"""
function sample_params()
	# first params of the autoencoder
	n_latents = sample(2:5)
	common_parameters = (n_latents=n_latents, init_seed = sample(1:Int(1e8)))

	zdim = sample(2 .^(1:6))
	hdims = linear_hdims(zdim, n_latents)
	par_vec = (2 .^(5:7), 2 .^(5:9), 2:5, 100:100:1000, vcat([0f0], 10f0 .^(-3:1)),
		10 .^range(-1f0, 2f0, length=10), 1:3, 10f0 .^(-4:-3),
		["relu", "swish", "tanh"])
	argnames = (:batchsize, :discriminator_hdim, :discriminator_nlayers, :nepochs, :λ, 
		:γ, :layer_depth, :lr,
		:activation, )
	autoencoder_parameters = merge((zdim = zdim,hdims = hdims), 
		(;zip(argnames, map(x->sample(x, 1)[1], par_vec))...))

	# parameters of the detectors
	detector_type = sample(["knn"])#, "ocsvm"])
	if detector_type == "knn"
		par_vec = 	(1:2:101, [:kappa, :gamma, :delta])
		argnames = (:k, :v)
	else
		nothing
	end
	detector_parameters = merge((detector_type=detector_type,),
		(;zip(argnames, map(x->sample(x, 1)[1], par_vec))...))

	# parameters of the classifier
	par_vec = (vcat([0f0], 10f0 .^(-3:1)), 2 .^(5:7), 50:50:500, 0.1:0.05:0.5)
	argnames = (:λ, :batchsize, :nepochs, :val_ratio)
	classifier_parameters = (;zip(argnames, map(x->sample(x, 1)[1], par_vec))...)	

	return common_parameters, autoencoder_parameters, detector_parameters, 
		classifier_parameters
end


function GenerativeAD.edit_params(data, parameters)
	idim = size(data[1][1],1)
	# put the largest possible zdim where zdim < idim, the model tends to converge poorly if the latent dim is larger than idim
	autoencoder_parameters = parameters[2]
	while autoencoder_parameters.zdim >= idim
		zdims = 2 .^(1:6)
		zdim_new = sample(zdims[zdims .< idim])
		hdims = linear_hdims(zdim_new, parameters[1].n_latents)
		autoencoder_parameters = merge(autoencoder_parameters, (zdim=zdim_new, hdims=hdims))
	end
	# ensure that knn's k is smaller than the number of samples
	
	local detector_parameters = parameters[3]
	if detector_parameters.detector_type == "knn"
		ks = collect(1:2:101)
		detector_parameters = merge(detector_parameters, (k=sample(ks[ks.<size(data[1][1],2)]),))
	end

	parameters[1], autoencoder_parameters, detector_parameters, parameters[4]
end


#############š#DELETE THIS##########šš
max_seed = 5
dataset = "iris"
contamination = 0f0
seed = 1
try_counter = 0
max_tries = 10*max_seed
cont_string = (contamination == 0.0) ? "" : "_contamination-$contamination"
parameters = sample_params()
savepath = datadir("experiments/tabular$cont_string/$(modelname)/$(dataset)/seed=$(seed)")
mkpath(savepath)
# get data
data = GenerativeAD.load_data(dataset, seed=seed, contamination=contamination)
# edit parameters
parameters = GenerativeAD.edit_params(data, parameters)


"""
	fit(data, parameters)

This is the most important function - returns `training_info` and a tuple or a vector of tuples `(score_fun, final_parameters)`.
`training_info` contains additional information on the training process that should be saved, the same for all anomaly score functions.
Each element of the return vector contains a specific anomaly score function - there can be multiple for each trained model.
Final parameters is a named tuple of names and parameter values that are used for creation of the savefile name.
"""
function fit(data, parameters)
	# construct model - constructor should only accept kwargs
	common_parameters, autoencoder_parameters, detector_parameters, 
		classifier_parameters = parameters
	Random.seed!(common_parameters.init_seed)
	if  detector_parameters.detector_type == "knn"
		detector_constructor = HierarchicalAD.knn_constructor
	else
		error("Unknown detector type $(detector_parameters.detector_type)")
	end	
	model = HAD(
		common_parameters.n_latents, 
	    merge(autoencoder_parameters, (verb=false,)), 
	    fvlae_constructor, 
	    Base.structdiff(detector_parameters, (detector_type="",)), 
	    detector_constructor,
	    merge(classifier_parameters, (patience=10, n_candidates=100))
	)
	Random.seed!()

	# fit train data
	try
		global info, fit_t, _, _, _ = @timed fit!(model, data[1][1], data[2][1], data[2][2])
	catch e
		# return an empty array if fit fails so nothing is computed
		@info "Failed training due to \n$e"
		return (fit_t = NaN, history=nothing, npars=nothing, model=nothing), [] 
	end

	# construct return information - put e.g. the model structure here for generative models
	training_info = (
		fit_t = fit_t,
		npars = sum(map(p->length(p), Flux.params(model.autoencoder))) + 
			sum(map(p->length(p), Flux.params(model.classifier))),
		model = model
		)

	# now return the different scoring functions
	L = 100
	training_info, [((x -> predict(model, x)), merge(parameters[1], parameters[2]))]
end

####################################################################
################ THIS PART IS COMMON FOR ALL MODELS ################
# only execute this if run directly - so it can be included in other files
if abspath(PROGRAM_FILE) == @__FILE__
	# set a maximum for parameter sampling retries
	try_counter = 0
	max_tries = 10*max_seed
	cont_string = (contamination == 0.0) ? "" : "_contamination-$contamination"
	while try_counter < max_tries
	    parameters = sample_params()

	    for seed in 1:max_seed
			savepath = datadir("experiments/tabular$cont_string/$(modelname)/$(dataset)/seed=$(seed)")
			mkpath(savepath)

			# get data
			data = GenerativeAD.load_data(dataset, seed=seed, contamination=contamination)
						
			# edit parameters
			edited_parameters = GenerativeAD.edit_params(data, parameters)
			
			@info "Trying to fit $modelname on $dataset with parameters $(edited_parameters)..."
			@info "Train/validation/test splits: $(size(data[1][1], 2)) | $(size(data[2][1], 2)) | $(size(data[3][1], 2))"
			@info "Number of features: $(size(data[1][1], 1))"

			# check if a combination of parameters and seed alread exists
			if GenerativeAD.check_params(savepath, edited_parameters)
				# fit
				training_info, results = fit(data, edited_parameters)

				# save the model separately			
				if training_info.model != nothing
					tagsave(joinpath(savepath, savename("model", merge(edited_parameters[1], edited_parameters[2]), 
						"bson", digits=5)), 
						Dict("model"=>training_info.model,
							"fit_t"=>training_info.fit_t,
							"parameters"=>edited_parameters
							), safe = true)
					training_info = merge(training_info, (model = nothing,))
				end

				# here define what additional info should be saved together with parameters, scores, labels and predict times
				save_entries = merge(training_info, 
					(
						modelname = modelname, 
						seed = seed, 
						dataset = dataset, 
						contamination = contamination,
						parameters = edited_parameters))

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
	(try_counter == max_tries) ? (@info "Reached $(max_tries) tries, giving up.") : nothing
end
