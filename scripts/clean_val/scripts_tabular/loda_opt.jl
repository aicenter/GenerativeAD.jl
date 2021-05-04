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
    "contamination"
    	arg_type = Float64
    	help = "contamination rate of training data"
    	default = 0.0
end
parsed_args = parse_args(ARGS, s)
@unpack dataset, max_seed, contamination = parsed_args

#######################################################################################
################ THIS PART IS TO BE PROVIDED FOR EACH MODEL SEPARATELY ################
modelname = "loda_opt"
function fit(data)
	# fit train data
	try
		global model, fit_t, _, _, _ = @timed GenerativeAD.Models.Loda(data[1][1])
	catch e
		# return an empty array if fit fails so nothing is computed
		return (fit_t = NaN,), [] 
	end

	# construct return information - put e.g. the model structure here for generative models

	training_info = (
		fit_t = fit_t,
		n_histograms = length(model.histograms),
		n_bins = [length(x.p) for x in model.histograms],
		model = nothing
		)

	# now return the different scoring functions
	training_info, [(x->model(x), (n_histograms=training_info.n_histograms,))]
end

####################################################################
################ THIS PART IS COMMON FOR ALL MODELS ################
# set a maximum for parameter sampling retries
try_counter = 0
max_tries = 10*max_seed
cont_string = (contamination == 0.0) ? "" : "_contamination-$contamination"
for seed in 1:max_seed
	savepath = datadir("experiments/tabular_clean_val_default/$(modelname)/$(dataset)/seed=$(seed)")
	mkpath(savepath)

	# get data
	data = GenerativeAD.load_data(dataset, seed=seed, contamination=contamination)
	
	@info "Trying to fit $modelname on $dataset..."

	# fit
	training_info, results = fit(data)

	# here define what additional info should be saved together with parameters, scores, labels and predict times
	save_entries = merge(training_info, (modelname = modelname, seed = seed, dataset = dataset, contamination = contamination))

	# now loop over all anomaly score funs
	for result in results
		GenerativeAD.experiment(result..., data, savepath; save_entries...)
	end
end
