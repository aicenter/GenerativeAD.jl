using DrWatson
@quickactivate
using ArgParse
using GenerativeAD
using StatsBase: fit!, predict, sample
using BSON

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

modelname = "RealNVP"
function sample_params()
	par_vec = ([2, 5, 10], [10, 50, 100, 200, 500, 1000], 2:3, [1f-4], [100], [30], [1f-6])
	argnames = (:nflows, :hsize, :nlayers, :lr, :batchsize, :patience, :wreg)

	return (;zip(argnames, map(x->sample(x, 1)[1], par_vec))...)
end

function fit(data, parameters)
	D = size(data[1][1], 1)

	model = GenerativeAD.Models.RealNVPFlow(
				parameters.nflows,
				D,
				parameters.hsize,
				parameters.nlayers)

	try
		global info, fit_t, _, _, _ = @timed fit!(model, data, parameters)
 	catch e
		@info "Failed training due to \n$e"
		return (fit_t = NaN,), []
	end

	training_info = (
		fit_t = fit_t,
		history = info.history,
		model = nothing
		)

	training_info, [(x -> predict(info.model, x), parameters)]
end


try_counter = 0
max_tries = 10*max_seed
while try_counter < max_tries
    parameters = sample_params()

    for seed in 1:max_seed
		savepath = datadir("experiments/tabular/$(modelname)/$(dataset)/seed=$(seed)")
		mkpath(savepath)

		data = GenerativeAD.load_data(dataset, seed=seed)
		edited_parameters = GenerativeAD.edit_params(data, parameters)

		if GenerativeAD.check_params(savepath, data, edited_parameters)
			@info "Started training $(modelname)$(edited_parameters) on $(dataset):$(seed)"
			@info "Train/valdiation/test splits: $(size(data[1][1], 2)) | $(size(data[2][1], 2)) | $(size(data[2][1], 2))"
			@info "Number of features: $(size(data[1][1], 1))"
			
			training_info, results = fit(data, edited_parameters)
			save_entries = merge(training_info, (modelname = modelname, seed = seed, dataset = dataset))

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
