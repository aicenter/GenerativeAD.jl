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
end
parsed_args = parse_args(ARGS, s)
@unpack dataset, max_seed, anomaly_classes = parsed_args

modelname ="svdd"

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
		["soft-boundary", "one-class"],
		[0.01, 0.1, 0.5, 0.99], # paper 0.1
		1:Int(1e8)
	)
	argnames = (
		:zdim, 
		:lr_ae, 
		:lr_svdd, 
		:batch_size, 
		:activation, 
		:objective,
		:nu,
		:init_seed
	)
	parameters = (;zip(argnames, map(x->sample(x, 1)[1], par_vec))...)
	return merge(parameters, (nlayers=nlayers, kernelsizes=kernelsizes,
		channels=channels, scalings=scalings))
end


function fit(data, parameters)
	# define models (Generator, Discriminator)
	parameters = merge(
		parameters, 
		(
			idim=size(data[1][1])[1:3], 
			batchnorm=true,
			decay=1e-6, # from paper
			iters=5000, 
			check_every=30, 
			patience=10, 
			ae_iters=10000
		)
	)
	svdd = GenerativeAD.Models.conv_ae_constructor(;parameters...) |> gpu

	# define optimiser
	try
		global info, fit_t, _, _, _ = @timed fit!(svdd, data, parameters)
	catch e
		println("Error caught => $(e).")
		return (fit_t = NaN, model = nothing, history = nothing, n_parameters = NaN), []
	end

	training_info = (
		fit_t = fit_t,
		model = info[2]|>cpu,
		history_svdd = info[1][1], # losses through time
		history_ae = info[1][2],
		npars = info[3], # number of parameters
		iters = info[4] # optim iterations of model
		)

	return training_info, [(x -> GenerativeAD.Models.anomaly_score_gpu(svdd, x), parameters)]
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

			# here, check if a model with the same parameters was already tested
			if GenerativeAD.check_params(savepath, parameters)
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
