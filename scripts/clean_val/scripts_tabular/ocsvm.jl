using DrWatson
@quickactivate
using ArgParse
using GenerativeAD
import StatsBase: fit!, predict
using StatsBase
using BSON
using Distances
using Statistics

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
    "contamination"
		arg_type = Float64
    	help = "contamination rate of training data"
    	default = 0.0
end
parsed_args = parse_args(ARGS, s)
@unpack dataset, max_seed, contamination = parsed_args

dataset = "iris"
max_seed = 2
contamination = 0.0
seed = 1
data = GenerativeAD.load_data(dataset, seed=seed, contamination=contamination)


#######################################################################################
################ THIS PART IS TO BE PROVIDED FOR EACH MODEL SEPARATELY ################
modelname = "ocsvm"
function median_l2_dist(X)
	dists = pairwise(Euclidean(), X)
	# take only the upper diagonal
	ds = []
	for i in 1:size(dists,1)
		for j in (i+1):size(dists,1)
			push!(ds, dists[i,j])
		end
	end
	ml=median(ds)
end
function set_params(data)
	D, N = size(data[1][1])
	gamma = 1/median_l2_dist(data[1][1])
	return (gamma=gamma, kernel="rbf", nu=0.5)
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
while try_counter < max_tries
    for seed in 1:max_seed
		savepath = datadir("experiments/tabular_clean_val_default/$(modelname)/$(dataset)/seed=$(seed)")
		mkpath(savepath)

		# get data
		data = GenerativeAD.load_data(dataset, seed=seed, contamination=contamination)
		
		# edit parameters
		parameters = set_params(data)
		
		@info "Trying to fit $modelname on $dataset with parameters $(parameters)..."
		# check if a combination of parameters and seed alread exists
		if GenerativeAD.check_params(savepath, parameters)
			# fit
			training_info, results = fit(data, parameters)
			# here define what additional info should be saved together with parameters, scores, labels and predict times
			save_entries = merge(training_info, (modelname = modelname, seed = seed, dataset = dataset, contamination = contamination))

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

