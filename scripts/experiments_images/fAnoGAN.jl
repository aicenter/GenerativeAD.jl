using DrWatson
@quickactivate
using ArgParse
using GenerativeAD
import StatsBase: fit!, predict
using StatsBase
using BSON
using Flux
using GenerativeModels

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
end
parsed_args = parse_args(ARGS, s)
@unpack dataset, max_seed, anomaly_classes = parsed_args


modelname = "fAnoGAN"

"""
	sample_params()

Should return a named tuple that contains a sample of model parameters.
"""
function sample_params()
	# first sample the number of layers
	nlayers = rand(2:4)
	kernelsizes = reverse((3,5,7,9)[1:nlayers])
	channels = reverse((16,32,64,128)[1:nlayers])
	scalings = reverse((1,2,2,2)[1:nlayers])
	
	par_vec = (
		2 .^(3:8), 
		10f0 .^(-4:-3),
		10f0 .^(-4:-3), 
		2 .^ (5:7), 
		["relu", "swish", "tanh"], 
		1:Int(1e8),
		[0.001,0.005,0.01,0.05,0.1]
	)
	argnames = (
		:zdim, 
		:lr_gan,
		:lr_enc, 
		:batchsize, 
		:activation, 
		:init_seed, 
		:weight_clip,
	)
	parameters = (;zip(argnames, map(x->sample(x, 1)[1], par_vec))...)
	return merge(parameters, (nlayers=nlayers, kernelsizes=kernelsizes,
		channels=channels, scalings=scalings))
end


"""
	fit(data, parameters)

This is the most important function - returns `training_info` and a tuple or a vector of tuples `(score_fun, final_parameters)`.
`training_info` contains additional information on the training process that should be saved, the same for all anomaly score functions.
Each element of the return vector contains a specific anomaly score function - there can be multiple for each trained model.
Final parameters is a named tuple of names and parameter values that are used for creation of the savefile name.
"""
function fit(data, parameters)
	# construct model - constructor should only accept kwargs
	default_params = (idim = size(data[1][1])[1:3], kappa = 1f0, weight_clip=0.1, lr_gan = 0.00005, lr_enc = 0.001, batch_size=128, 
		patience=10, check_every=30, iters=10000, mtt=82800, n_critic=1, usegpu=true)
	

	# construct model - constructor should only accept kwargs
	model = GenerativeAD.Models.fanogan_constructor(;idim = size(data[1][1])[1:3], parameters...) |> gpu

	max_iter = 50 # this should be enough

	# fit train data
	
	try
		global info, fit_t, _, _, _ = @timed fit!(model, data, merge(default_params, parameters))
	catch e
		# return an empty array if fit fails so nothing is computed
		@info "Failed training due to \n$e"
		return (fit_t = NaN, history=nothing, npars=nothing, model=nothing), [] 
	end
	model = info[1]
	
	# construct return information - put e.g. the model structure here for generative models
	training_info = (
		fit_t = fit_t,
		history = info[2],
		npars = info[3],
		model = model |> cpu
		)

	# now return the different scoring functions
	training_info, [(x -> GenerativeAD.Models.anomaly_score_gpu(model, x), parameters)]
end


___________________________________________________________________________________________________________________
try_counter = 0
max_tries = 10*max_seed

while try_counter < max_tries
	parameters = sample_params()

	for seed in 1:max_seed
		for i in 1:anomaly_classes
			savepath = datadir("experiments/images/$(modelname)/$(dataset)/ac=$(i)/seed=$(seed)")
			mkpath(savepath)

			data = GenerativeAD.load_data(dataset, seed=seed, anomaly_class_ind=i)

			@info "Trying to fit $modelname on $dataset with parameters $(parameters)..."
			@info "Train/validation/test splits: $(size(data[1][1], 4)) | $(size(data[2][1], 4)) | $(size(data[3][1], 4))"
			@info "Number of features: $(size(data[1][1])[1:3])"

			# here, check if a model with the same parameters was already tested
			if GenerativeAD.check_params(savepath, parameters) & GenerativeAD.Models.check_scaling(size(data[1][1])[1:3], parameters.scalings) 
				# fit model
				training_info, results = fit(data, parameters)
				# saving model separately
				if training_info.model !== nothing
					tagsave(joinpath(savepath, savename("model", parameters, "bson", digits=5)), Dict("model"=>training_info.model), safe = true)
					training_info = merge(training_info, (model = nothing,))
				end
				save_entries = merge(training_info, (modelname = modelname, seed = seed, dataset = dataset, anomaly_class = i))
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
(try_counter == max_tries) ? (@info "Reached $(max_tries) tries, giving up.") : nothing

