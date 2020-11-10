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
end
parsed_args = parse_args(ARGS, s)
@unpack dataset, max_seed = parsed_args

modelname ="adVAE"

function sample_params()
	argnames = (
		:hdim, 
		:zdim, 
		:nlayers, 
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
		2 .^(4:9),
		2 .^(1:8),
		3:4,
		["relu", "swish", "tanh"],
		[0.0005, 0.001, 0.005], # γ
		[0.005, 0.01, 0.05], # λ  # better params
		[1, 1.5],  #mx #0.5:0.5:2.5
 		[40, 50, 60], #mz 10:10:100
		10f0 .^ (-4:-3),
		0f0:0.1:0.5,
		2 .^ (5:6),
		[10000],
		[30],
		[10],
		1:Int(1e8),
	)

	return NamedTuple{argnames}(map(x->sample(x,1)[1], par_vec))
end

function fit(data, parameters)
	# define models (Generator, Discriminator)
	advae = GenerativeAD.Models.adVAE(;parameters...)

	# define optimiser
	try
		global info, fit_t, _, _, _ = @timed fit!(advae |> gpu, data, parameters)
	catch e
		println("Error caught.")
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
	[(x -> GenerativeAD.Models.anomaly_score(advae|>cpu, x; dims=1, L=100)[:], merge(parameters, (L=100, )))]
	# L = samples for one x in anomaly_score computation
end

#_________________________________________________________________________________________________

try_counter = 0
max_tries = 10*max_seed

while try_counter < max_tries
	parameters = sample_params()

	for seed in 1:max_seed
		savepath = datadir("experiments/tabular/$(modelname)/$(dataset)/seed=$(seed)")

		data = GenerativeAD.load_data(dataset, seed=seed)
		# update parameter
		parameters = merge(parameters, (idim=size(data[1][1],1), ))
		# here, check if a model with the same parameters was already tested
		if GenerativeAD.check_params(savepath, parameters)
			#(X_train,_), (X_val, y_val), (X_test, y_test) = data
			training_info, results = fit(data, parameters)
			# saving model separately
			if training_info.model !== nothing
				tagsave(joinpath(savepath, savename("model", parameters, "bson", digits=5)), Dict("model"=>training_info.model), safe = true)
				training_info = merge(training_info, (model = nothing,))
			end
			save_entries = merge(training_info, (modelname = modelname, seed = seed, dataset = dataset))
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
