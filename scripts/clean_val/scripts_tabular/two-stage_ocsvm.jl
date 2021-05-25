using DrWatson
@quickactivate
using ArgParse
using GenerativeAD
using GenerativeAD.Models
import StatsBase: fit!, predict
using StatsBase
using BSON
# because of vae and 2stage
using DataFrames
using CSV
using ValueHistories
using Flux
using ConditionalDists
using GenerativeModels
import GenerativeModels: VAE
using Distributions
using DistributionsAD
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

# we select the encoding of the model with lowest loss
tab_name = "vae_LOSS_tabular"
mi = 2

#######################################################################################
################ THIS PART IS TO BE PROVIDED FOR EACH MODEL SEPARATELY ################
# get criterion
sp = split(tab_name, "_")
enc = sp[1]
criterion = lowercase(sp[2])

modelname = "$(enc)_ocsvm"

# default hyperparams
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

function fit(data, parameters, aux_info)
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
	training_info, [(x->predict(model, x), merge(parameters, aux_info))]
end

####################################################################
################ THIS PART IS COMMON FOR ALL MODELS ################
# set a maximum for parameter sampling retries
cont_string = (contamination == 0.0) ? "" : "_contamination-$contamination"
for seed in 1:max_seed
	savepath = datadir("experiments/tabular_clean_val_default/$(modelname)/$(dataset)/seed=$(seed)")
	mkpath(savepath)

	aux_info = (model_index=mi, criterion=criterion)
	# get data
	data = GenerativeAD.load_data(dataset, seed=seed, contamination=contamination)
	output, encode_t, _, _, _ = @timed GenerativeAD.Models.load_encoding(tab_name, data, dataset=dataset, seed=seed, model_index=mi)
	data, encoding_name, encoder_params, encoder_fit_t = output
		
	# set parameters
	parameters = set_params(data)
	
	@info "Trying to fit $modelname on $dataset with parameters $(parameters)..."
	# check if a combination of parameters and seed alread exists
	# fit
	training_info, results = fit(data, parameters, aux_info)
	# here define what additional info should be saved together with parameters, scores, labels and predict times
	save_entries = merge(training_info, (modelname = modelname, 
										 seed = seed, 
										 dataset = dataset, 
										 encoder=encoding_name,
										 encoder_fit_t = encoder_fit_t,
										 encoder_params=encoder_params, 
										 encode_t = encode_t,
										 model_index=mi,
										 criterion=criterion,
										 contamination = contamination))

	# now loop over all anomaly score funs
	for result in results
		GenerativeAD.experiment(result..., data, savepath; save_entries...)
	end
end

