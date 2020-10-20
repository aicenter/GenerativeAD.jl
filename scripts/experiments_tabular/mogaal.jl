using DrWatson
@quickactivate
using ArgParse
using GenerativeAD
import StatsBase: fit!, predict
using StatsBase
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

modelname = "MO_GAAL"

function sample_params()
	argnames = (
		:k, 
		:stop_epochs, 
		:lr_d, 
		:lr_g, 
		:decay, 
		:momentum, 
		:contamination, 
	)
	par_vec = (
		5:20,
		10:10:100,
		10f0 .^ (-4:-2),
		10f0 .^ (-4:-2),
		[1e-6],
		0.5:0.1:0.9,
		0.1:0.1:0.5,
	)
	return merge(NamedTuple{argnames}(map(x->sample(x,1)[1], par_vec)))
end

function fit(data, parameters)
	# construct model - constructor should only accept kwargs
	model = GenerativeAD.Models.MO_GAAL(;parameters...)
 
	# fit train data
	try
		global info, fit_t, _, _, _ = @timed fit!(model, data[1][1])
	catch e
		println("error caught", e)
		# return an empty array if fit fails so nothing is computed
		return (fit_t = NaN,), [] 
	end

	# construct return information - put e.g. the model structure here for generative models
	training_info = (
		fit_t = fit_t,
		model = nothing,
		npars = parameters.k * 2 * (size(data[1][1],1)^2) 
			+ Int(ceil(sqrt(size(data[1][1],2)))) * size(data[1][1],1) 
			+ Int(ceil(sqrt(size(data[1][1],2))))
		)

	# now return the different scoring functions
	training_info, [(x->predict(model, x), parameters)]
end


try_counter = 0
max_tries = 10*max_seed
while try_counter < max_tries
	parameters = sample_params()

	for seed in 1:max_seed
		savepath = datadir("experiments/tabular/$(modelname)/$(dataset)/seed=$(seed)")
		mkpath(savepath)

		# get data
		data = GenerativeAD.load_data(dataset, seed=seed)
		
		# edit parameters
		edited_parameters = GenerativeAD.edit_params(data, parameters)

		@info "Trying to fit $modelname on $dataset with parameters $(edited_parameters)..."
		# check if a combination of parameters and seed alread exists
		if GenerativeAD.check_params(savepath, edited_parameters)
			# fit
			training_info, results = fit(data, edited_parameters)
			# here define what additional info should be saved together with parameters, scores, labels and predict times
			save_entries = merge(training_info, (modelname = modelname, seed = seed, dataset = dataset))

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
(try_counter == max_tries) ? (@info "Reached $(max_tries) tries, giving up.") : nothing

