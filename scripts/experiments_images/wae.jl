using DrWatson
@quickactivate
using ArgParse
using GenerativeAD
import StatsBase: fit!, predict
using StatsBase
using BSON
using Flux
using IPMeasures
using GenerativeModels
using Distributions

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
end
parsed_args = parse_args(ARGS, s)
@unpack dataset, max_seed, anomaly_classes, method = parsed_args

#######################################################################################
################ THIS PART IS TO BE PROVIDED FOR EACH MODEL SEPARATELY ################
modelname = "wae"

DrWatson.projectdir() = "/home/skvarvit/generativead/GenerativeAD.jl"

# sample parameters, should return a Dict of model kwargs 
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
	
	par_vec = (2 .^(3:8), 10f0 .^(-4:-3), 2 .^ (5:7), ["relu", "swish", "tanh"], 1:Int(1e8),
				["imq", "gauss", "rq"], 10f0 .^ (-3:0), 10f0 .^(-1:0), 2 .^ (1:6))
	argnames = (:zdim, :lr, :batchsize, :activation, :init_seed, :kernel, :sigma, :lambda,
		:k)
	parameters = (;zip(argnames, map(x->sample(x, 1)[1], par_vec))...)
	return merge(parameters, (nlayers=nlayers, kernelsizes=kernelsizes,
		channels=channels, scalings=scalings))
end
batch_score(scoref, model, x, batchsize=512) =
	vcat(map(y->cpu(scoref(model, gpu(Array(y)))), Flux.Data.DataLoader(x, batchsize=batchsize))...)
"""
	fit(data, parameters)

This is the most important function - returns `training_info` and a tuple or a vector of tuples `(score_fun, final_parameters)`.
`training_info` contains additional information on the training process that should be saved, the same for all anomaly score functions.
Each element of the return vector contains a specific anomaly score function - there can be multiple for each trained model.
Final parameters is a named tuple of names and parameter values that are used for creation of the savefile name.
"""
function fit(data, parameters)
	# first construct the VAMP pseudoinput array
	X = data[1][1]
	pseudoinput_mean = mean(X, dims=ndims(X))

	# construct model - constructor should only accept kwargs
	idim = size(X)[1:3]

	# construct model - constructor should only accept kwargs
	model = GenerativeAD.Models.conv_vae_constructor(;idim=idim, prior="vamp", 
		pseudoinput_mean=pseudoinput_mean, parameters...) |> gpu

	# construct loss function
	if parameters.kernel == "imq"
		k = IMQKernel(parameters.sigma)
	elseif parameters.kernel == "gauss"
		k = GaussianKernel(parameters.sigma)
	elseif parameters.kernel == "rq"
		k = RQKernel(parameters.sigma)
	else
		error("given kernel not known")
	end
	loss(m::GenerativeModels.VAE,x) = parameters.lambda*mmd_mean(m, gpu(Array(x)), k) .- 
		mean(logpdf(m.decoder, gpu(Array(x)), rand(m.encoder, gpu(Array(x)))))
	loss(m::GenerativeModels.VAE, x, batchsize::Int) = 
		mean(map(y->loss(m,y), Flux.Data.DataLoader(x, batchsize=batchsize)))

	# fit train data
	try
		global info, fit_t, _, _, _ = @timed fit!(model, data, loss; max_iters = 20000, 
			max_train_time=23*3600/max_seed/anomaly_classes/4, 
			patience=10, check_interval=50, parameters...)
	catch e
		# return an empty array if fit fails so nothing is computed
		@info "Failed training due to \n$e"
		return (fit_t = NaN, history=nothing, npars=nothing, model=nothing), [] 
	end
	model = info.model
	
	# produce encodings
	if model != nothing
		encodings = map(x->cpu(GenerativeAD.Models.encode_mean_gpu(model, x, 128)), (data[1][1], data[2][1], data[3][1]))
	else
		encodings = (nothing, nothing, nothing)
	end

	# construct return information - put e.g. the model structure here for generative models
	training_info = (
		fit_t = fit_t,
		history = info.history,
		npars = info.npars,
		model = model |> cpu,
		tr_encodings = encodings[1],
		val_encodings = encodings[2],
		tst_encodings = encodings[3]
		)

	# now return the different scoring functions
	training_info, [
		(x -> batch_score(GenerativeAD.Models.reconstruction_score, model, x), merge(parameters, (score = "reconstruction",))),
		(x -> batch_score(GenerativeAD.Models.reconstruction_score_mean, model, x), merge(parameters, (score = "reconstruction-mean",))),
		]
end

####################################################################
################ THIS PART IS COMMON FOR ALL MODELS ################
# only execute this if run directly - so it can be included in other files
if abspath(PROGRAM_FILE) == @__FILE__
	# set a maximum for parameter sampling retries
	try_counter = 0
	max_tries = 10*max_seed
	while try_counter < max_tries
		parameters = sample_params()

		for seed in 1:max_seed
			for i in 1:anomaly_classes
				savepath = datadir("experiments/images_$(method)/$(modelname)/$(dataset)/ac=$(i)/seed=$(seed)")
				mkpath(savepath)

				# get data
				data = GenerativeAD.load_data(dataset, seed=seed, anomaly_class_ind=i, method=method)
				
				# edit parameters
				edited_parameters = GenerativeAD.edit_params(data, parameters)

				@info "Trying to fit $modelname on $dataset with parameters $(edited_parameters)..."
				@info "Train/validation/test splits: $(size(data[1][1], 4)) | $(size(data[2][1], 4)) | $(size(data[3][1], 4))"
				@info "Number of features: $(size(data[1][1])[1:3])"

				# check if a combination of parameters and seed alread exists
				if GenerativeAD.check_params(savepath, edited_parameters)
					# fit
					training_info, results = fit(data, edited_parameters)

					# save the model separately			
					if training_info.model != nothing
						tagsave(joinpath(savepath, savename("model", edited_parameters, "bson", digits=5)), 
							Dict("model"=>training_info.model,
								 "tr_encodings"=>training_info.tr_encodings,
								 "val_encodings"=>training_info.val_encodings,
								 "tst_encodings"=>training_info.tst_encodings,
								 "fit_t"=>training_info.fit_t,
								 "history"=>training_info.history,
								 "parameters"=>edited_parameters
								 ), 
							safe = true)
						training_info = merge(training_info, 
							(model=nothing,tr_encodings=nothing,val_encodings=nothing,tst_encodings=nothing))
					end

					# here define what additional info should be saved together with parameters, scores, labels and predict times
					save_entries = merge(training_info, (modelname = modelname, seed = seed, dataset = dataset, anomaly_class = i))

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
end
