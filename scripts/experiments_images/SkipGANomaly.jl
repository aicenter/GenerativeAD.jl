using DrWatson
@quickactivate
using ArgParse
using GenerativeAD
using GenerativeAD.Models: anomaly_score, generalized_anomaly_score_gpu
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

modelname = "Conv-SkipGANomaly"

function sample_params()
	argnames = (:num_filters, :extra_layers, :lr, :batch_size,
				:iters, :check_every, :patience, :lambda, :init_seed, )
	par_vec = (
			   2 .^ (3:6),
			   [0:3 ...],
			   10f0 .^ (-4:-3),
			   2 .^ (5:7),
			   [10000],
			   [30],
			   [10],
			   [0.9],
			   1:Int(1e8),
			   )
	w = (weights= sample([1,10:10:90...],3),)
	return merge(NamedTuple{argnames}(map(x->sample(x,1)[1], par_vec)), w)
end

"""
	function fit(data, parameters)

parameters => type named tuple with keys
	num_filters   - number of kernels/masks in convolutional layers
	extra_layers  - number of additional conv layers in discriminator
	lr            - learning rate for optimiser
	iters         - number of optimisation steps (iterations) during training
	batch_size    - batch/minibatch size

Note:
	data = load_data("MNIST")
	(x_train, y_train), (x_val, y_val), (x_test, y_test) = data
"""
function fit(data, parameters)
	# define models (Generator, Discriminator)
	model, _ = GenerativeAD.Models.SkipGANomaly_constructor(parameters)

	try
		global info, fit_t, _, _, _ = @timed fit!(model |>gpu , data, parameters)
	catch e
		println("Error caught.")
		return (fit_t = NaN, model = nothing, history = nothing, n_parameters = NaN), []
	end

	training_info = (
		fit_t = fit_t,
		model = (info[2] |> cpu),
		history = info[1], # losses through time
		npars = info[3], # number of parameters
		iters = info[4] # optim iterations of model
		)


	return training_info, [(x -> generalized_anomaly_score_gpu(model|>cpu, x, R=r, L=l, lambda=lam), 
		merge(parameters, (R=r, L=l, test_lambda=lam,)))
		for r in ["mae", "mse"] for l in ["mae", "mse"] for lam = 0.1:0.1:0.9 ]
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

			isize = (isize % 32 != 0) ? isize + 32 - isize % 32 : isize
			# update parameter
			parameters = merge(parameters, (isize=isize, in_ch = in_ch, out_ch = 1))
			# here, check if a model with the same parameters was already tested
			if GenerativeAD.check_params(savepath, parameters)

				data = GenerativeAD.Models.preprocess_images(data, parameters, denominator=32)
				#(X_train,_), (X_val, y_val), (X_test, y_test) = data
				training_info, results = fit(data, parameters)
				# saving model separately
				if training_info.model != nothing
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
