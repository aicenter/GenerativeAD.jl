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
end
parsed_args = parse_args(ARGS, s)
@unpack dataset, max_seed, anomaly_classes = parsed_args

modelname = "Conv-GANomaly"


function sample_params()
	argnames = (:hdim, :num_filters, :extra_layers, :lr, :batch_size,
				:iters, :check_every, :patience, )
	par_vec = (
			2 .^(3:8),
			2 .^(2:8)],
			[0:3...],
			10f0 .^(-4:-3), #[0.0001:0.0001:0.001...],
			 2 .^ (5:7),
			[10000],
			[30],
			[10],
			)
	return NamedTuple{argnames}(map(x->sample(x,1)[1], options))
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

	return training_info, [(x -> GenerativeAD.Models.anomaly_score_gpu(generator|>cpu, x; dims=3), parameters)]
end

#_________________________________________________________________________________________________

try_counter = 0
max_tries = 10*max_seed

while try_counter < max_tries
	parameters = sample_params()

	for seed in 1:max_seed
		for i in 1:anomaly_classes
			savepath = datadir("experiments/images/$(modelname)/$(dataset)/ac=$(i)/seed=$(seed)")

			data = GenerativeAD.load_data(dataset, seed=seed, anomaly_class_ind=i)
			# computing additional parameters
			in_ch = size(data[1][1],3)
			isize = maximum([size(data[1][1],1),size(data[1][1],2)])

			isize = isize + 16 - isize % 16
			# update parameter
			parameters = merge(parameters, (isize=isize, in_ch = in_ch, out_ch = 1))
			# here, check if a model with the same parameters was already tested
			if GenerativeAD.check_params(savepath, parameters, data)

				data = GenerativeAD.Models.preprocess_images(data, parameters)
				#(X_train,_), (X_val, y_val), (X_test, y_test) = data
				training_info, results = fit(data, parameters)
				# saving model separately
                if training_info.model != nothing
                    tagsave(joinpath(savepath, savename("model", parameters, "bson")), Dict("model"=>training_info.model), safe = true)
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
