using DrWatson
@quickactivate
using ArgParse
using GenerativeAD
using GenerativeAD.Models
using BSON
using StatsBase: fit!, predict, sample

using Flux
using MLDataPattern

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

modelname ="adVAE"

function sample_params()
	argnames = (
		:zdim,
		:nf, 
		:extra_layers,
		:activation,
		:gamma,
		:lambda,
		:mx,
		:mz,
		:lr, 
		:decay,
		:batch_size, 
		:iters, 
		:check_every, 
		:patience, 
		:init_seed,
	)
	par_vec = (
		2 .^(3:8), # dim of latent space and number neurons in transformer
		2 .^(2:7), # number of filters
		[0:2 ...], # extra layers
		["relu", "swish", "tanh"], # activation function for dense layers
		[0.0005, 0.001, 0.005], # γ
		[0.005, 0.01, 0.05], # λ  
		[1, 1.5], #mx
		[40, 50, 60], #mz
		10f0 .^ (-4:-3), # lr
		0f0:0.1:0.5, # weight decay
		2 .^ (5:6), # batch_size
		[10000],
		[30],
		[10],
		1:Int(1e8),
	)

	return NamedTuple{argnames}(map(x->sample(x,1)[1], par_vec))
end

function fit(data, parameters)
	# define models (Generator, Discriminator)
	advae = Conv_adVAE(;parameters...)

	# define optimiser
	try
		global info, fit_t, _, _, _ = @timed fit!(advae |> gpu, data, parameters)
	catch e
		println("Error caught => $(e).")
		return (fit_t = NaN, model = nothing, history = nothing, n_parameters = NaN), []
	end

	training_info = (
		fit_t = fit_t,
		model = info[2]|>cpu,
		history = info[1], # losses through time
		npars = info[3], # number of parameters
		iters = info[4] # optim iterations of model
		)

	return training_info, 
	[(x -> GenerativeAD.Models.anomaly_score(advae|>cpu, x; dims=(1,2,3), L=100)[:], merge(parameters, (L=100, )))]
	# L = samples for one x in anomaly_score computation
end

#_________________________________________________________________________________________________

try_counter = 0
max_tries = 10*max_seed
cont_string = (contamination == 0.0) ? "" : "_contamination-$contamination"
while try_counter < max_tries
	parameters = sample_params()

	for seed in 1:max_seed
		for i in 1:anomaly_classes
			savepath = datadir("experiments/images_$(method)$cont_string/$(modelname)/$(dataset)/ac=$(i)/seed=$(seed)")

			data = GenerativeAD.load_data(dataset, seed=seed, method=method, anomaly_class_ind=i)
			# computing additional parameters
			in_ch = size(data[1][1],3)
			isize = maximum([size(data[1][1],1),size(data[1][1],2)])

			isize = (isize % 16 != 0) ? isize + 16 - isize % 16 : isize
			# update parameter
			parameters = merge(parameters, (isize=isize, in_ch = in_ch, out_ch = 1))
			# here, check if a model with the same parameters was already tested
			if GenerativeAD.check_params(savepath, parameters)

				data = GenerativeAD.Models.preprocess_images(data, parameters)
				#(X_train,_), (X_val, y_val), (X_test, y_test) = data
				training_info, results = fit(data, parameters)
				# saving model separately
				if training_info.model !== nothing
					tagsave(joinpath(savepath, savename("model", parameters, "bson", digits=5)), Dict("model"=>training_info.model), safe = true)
					training_info = merge(training_info, (model = nothing,))
				end
				save_entries = merge(training_info, (modelname = modelname, seed = seed, dataset = dataset, anomaly_class = i,
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
end
(try_counter == max_tries) ? (@info "Reached $(max_tries) tries, giving up.") : nothing
