using DrWatson
@quickactivate
using ArgParse
using GenerativeAD
using GenerativeAD.Models: anomaly_score
using BSON
using StatsBase: fit!, predict, sample

using Flux
using MLDataPattern

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
@unpack category, max_seed, contamination = parsed_args

modelname = "Conv-GANomaly"


function sample_params()
	argnames = (:hdim, :num_filters, :extra_layers, :lr, :batch_size,
				:iters, :check_every, :patience, :init_seed, )
	par_vec = (
			2 .^(3:8),
			2 .^(2:7),
			[0:1 ...],
			10f0 .^ (-4:-3),
			2 .^ (5:7),
			[10000],
			[30],
			[10],
			1:Int(1e8),
			)
	return NamedTuple{argnames}(map(x->sample(x,1)[1], par_vec))
end

"""
	function fit(data, parameters)

parameters => type named tuple with keys
	latent_dim    - dimension of latent space on the encoder's end
	num_filters   - number of kernels/masks in convolutional layers
	extra_layers  - number of additional conv layers
	lr            - learning rate for optimiser
	iters         - number of optimisation steps (iterations) during training
	batch_size    - batch/minibatch size

Note:
	data = load_data("MNIST")
	(x_train, y_train), (x_val, y_val), (x_test, y_test) = data
"""
function fit(data, parameters)
	# define models (Generator, Discriminator)
	generator, discriminator, _, _ = GenerativeAD.Models.ganomaly_constructor(parameters)

	# define optimiser
	try
		global info, fit_t, _, _, _ = @timed fit!(generator|>gpu, discriminator|>gpu, data, parameters)
	catch e
		println("Error caught.")
		return (fit_t = NaN, model = nothing, history = nothing, n_parameters = NaN), []
	end

	training_info = (
		fit_t = fit_t,
		model = (info[2]|>cpu, info[3]|>cpu),
		history = info[1], # losses through time
		npars = info[4], # number of parameters
		iters = info[5] # optim iterations of model
		)

	return training_info, [(x -> GenerativeAD.Models.anomaly_score_gpu(generator|>cpu, x; dims=3)[:], parameters)]
end

#_________________________________________________________________________________________________

if abspath(PROGRAM_FILE) == @__FILE__
	try_counter = 0
	max_tries = 10*max_seed
	cont_string = (contamination == 0.0) ? "" : "_contamination-$contamination"
	while try_counter < max_tries
		parameters = sample_params()

		for seed in 1:max_seed
			savepath = datadir("experiments/images_mvtec$cont_string/$(modelname)/$(category)/ac=1/seed=$(seed)")
			mkpath(savepath)

			# get data
			data = GenerativeAD.load_data("MVTec-AD", seed=seed, category=category, 
					contamination=contamination, img_size=128)

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
				#fit
				training_info, results = fit(data, parameters)

				# saving model separately
				if training_info.model !== nothing
					tagsave(joinpath(savepath, savename("model", parameters, "bson", digits=5)), 
						Dict("model"=>training_info.model), 
						safe = true)
					training_info = merge(training_info, (model = nothing,))
				end
				save_entries = merge(training_info, (modelname = modelname, seed = seed,
					category = category,
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
end