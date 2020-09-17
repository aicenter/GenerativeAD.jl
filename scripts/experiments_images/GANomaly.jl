using DrWatson
@quickactivate
using ArgParse
using GenerativeAD
using GenerativeAD.Models: anomaly_score
using BSON
using StatsBase: fit!, predict

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
	argnames = (:latent_dim, :num_filters, :extra_layers, :lr, :epochs, :batch_size, :patience, )
	options = (
			[10:10:200...],
			[2^x for x=2:8],
			[1:5...],
			[0.0001:0.0001:0.001..., 0.002:0.001:0.01...],
			[20],
			[2^x for x=2:8],
            [3], 
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
		return (fit_t = NaN,), []
	end

	training_info = (
		fit_t = fit_t,
		model = (generator|>cpu, discriminator|>cpu),
		history = info[1] # losses through time
		)

	return training_info, [(x -> GenerativeAD.Models.anomaly_score(generator|>cpu, x; dims=3), parameters)]
	# not sure if I should return generator and disciriminator in GPU
end

#_________________________________________________________________________________________________


try_counter = 0
max_tries = 10*max_seed

while try_counter < max_tries
	parameters = sample_params()

    for seed in 1:max_seed
        for i in 1:anomaly_classes
            savepath = datadir("experiments/images/$(modelname)/$(dataset)_ac=$(i)/seed=$(seed)")

            data = GenerativeAD.load_data(dataset, seed=seed, anomaly_class_ind=i)
        	# computing additional parameters
        	in_ch = size(data[1][1],3)
        	isize = maximum([size(data[1][1],1),size(data[1][1],2)])

        	isize = isize + 16 - isize % 16
        	# update parameter
        	parameters = merge(parameters, (isize=isize, in_ch = in_ch, out_ch = 1))
        	# here, check if a model with the same parameters was already tested
        	if GenerativeAD.check_params(savepath, parameters, data)

        		data = GenerativeAD.preprocess_images(data, parameters)
        		#(X_train,_), (X_val, y_val), (X_test, y_test) = data
                training_info, results = fit(data, parameters)

        		save_entries = merge(training_info, (modelname = modelname, seed = seed, dataset = dataset, anomaly_class = i))

        		# now loop over all anomaly score funs
        		for result in results
        			GenerativeAD.experiment(result..., data, savepath; save_entries...)
        		end
        		break
        	else
        		@info "Model already present, sampling new hyperparameters..."
        		global try_counter += 1
            end
        end
    end
end
(try_counter == max_tries) ? (@info "Reached $(max_tries) tries, giving up.") : nothing
