using DrWatson
@quickactivate
using ArgParse
using GenerativeAD
import StatsBase: fit!, predict
using StatsBase
using BSON
using PyCall

s = ArgParseSettings()
@add_arg_table! s begin
	"max_seed"
		default = 1
		arg_type = Int
		help = "maximum number of seeds to run through"
	"dataset"
		default = "iris"
		arg_type = String
		help = "dataset"
	"sampling"
		default = "random"
		arg_type = String
		help = "sampling of hyperparameters"
    "contamination"
    	arg_type = Float64
    	help = "contamination rate of training data"
    	default = 0.0
end
parsed_args = parse_args(ARGS, s)
@unpack dataset, max_seed, sampling, contamination = parsed_args

#######################################################################################
################ THIS PART IS TO BE PROVIDED FOR EACH MODEL SEPARATELY ################
modelname = "ocsvm"

parameter_rng = (
	gamma 		= round.([10^x for x in -4:0.1:2],digits=5),
	kernel 		= ["poly", "rbf", "sigmoid"],
	nu 			= [0.01, 0.5, 0.99]
)

function create_space()
	pyReal = pyimport("skopt.space")["Real"]
	pyInt = pyimport("skopt.space")["Integer"]
	pyCat = pyimport("skopt.space")["Categorical"]
	
	(;
		gamma 		= pyReal(1e-4, 1e2, prior="log-uniform",		name="gamma"),
		kernel 		= pyCat(categories=["poly", "rbf", "sigmoid"],	name="kernel"),
		nu 			= pyReal(0.01, 0.99, 							name="nu")
	)
end

function fit(data, parameters)
	# construct model - constructor should only accept kwargs
	model = GenerativeAD.Models.OCSVM(;parameters...)

	# sumbsample and fit train data
	tr_data = data[1][1]
	M,N = size(tr_data)
	tr_data = tr_data[:, sample(1:N, min(10000, N), replace=false)]
	try
		global info, fit_t, _, _, _ = @timed fit!(model, data[1][1])
	catch e
		# return an empty array if fit fails so nothing is computed
		return (fit_t = NaN,), [] 
	end

	# construct return information - put e.g. the model structure here for generative models
	training_info = (
		fit_t = fit_t,
		model = nothing
		)

	# now return the different scoring functions
	training_info, [(x->predict(model, x), parameters)]
end

####################################################################
################ THIS PART IS COMMON FOR ALL MODELS ################
# set a maximum for parameter sampling retries
try_counter = 0
max_tries = 10*max_seed
cont_string = (contamination == 0.0) ? "" : "_contamination-$contamination"
prefix = "experiments/tabular$(cont_string)"
while try_counter < max_tries
	if sampling == "bayes"
		parameters = GenerativeAD.bayes_params(
								create_space(), 
								datadir("$(prefix)/$(modelname)/$(dataset)"),
								parameter_rng)
	else
		parameters = GenerativeAD.sample_params(parameter_rng)
	end

    for seed in 1:max_seed
		savepath = datadir("experiments/tabular$cont_string/$(modelname)/$(dataset)/seed=$(seed)")
		mkpath(savepath)

		# get data
		data = GenerativeAD.load_data(dataset, seed=seed, contamination=contamination)
		
		# edit parameters
		edited_parameters = GenerativeAD.edit_params(data, parameters)
		
		@info "Trying to fit $modelname on $dataset with parameters $(edited_parameters)..."
		# check if a combination of parameters and seed alread exists
		if GenerativeAD.check_params(savepath, edited_parameters)
			# fit
			training_info, results = fit(data, edited_parameters)
			# here define what additional info should be saved together with parameters, scores, labels and predict times
			save_entries = merge(training_info, (modelname = modelname, seed = seed, dataset = dataset, contamination = contamination))

			# now loop over all anomaly score funs
			all_scores = [GenerativeAD.experiment(result..., data, savepath; save_entries...) for result in results]
			if sampling == "bayes" && length(all_scores) > 0
				@info("Updating cache with $(length(all_scores)) results.")
				GenerativeAD.update_bayes_cache(datadir("$(prefix)/$(modelname)/$(dataset)"), 
						[all_scores[1]]; ignore=Set([:init_seed])) # so far use only one score
			end
			global try_counter = max_tries + 1
		else
			@info "Model already present, trying new hyperparameters..."
			global try_counter += 1
		end
	end
end
(try_counter == max_tries) ? (@info "Reached $(max_tries) tries, giving up.") : nothing

