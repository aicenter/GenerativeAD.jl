include("utils.jl") # contains most dependencies and the saving function
using GenerativeAD.Models: anomaly_score, generalized_anomaly_score_gpu

s = ArgParseSettings()
@add_arg_table! s begin
	"dataset"
		default = "MNIST"
		arg_type = String
		help = "dataset"
end
parsed_args = parse_args(ARGS, s)
@unpack dataset = parsed_args

#######################################################################################
################ THIS PART IS TO BE PROVIDED FOR EACH MODEL SEPARATELY ################
modelname = "Conv-SkipGANomaly"

"""
This returns encodings, parameters and scoring functions in order to reconstruct the experiment. 
This is a slightly updated version of the original run script.
"""
function evaluate(model_data, data, parameters)
	# load the model file, extract params and model
	model = model_data["model"] |> gpu
	
	# construct return information - put e.g. the model structure here for generative models
	training_info = (
		fit_t = get(model_data, "fit_t", nothing),
		history = get(model_data, "history", nothing),
		iters = get(model_data, "iters", nothing),
		npars = get(model_data, "npars", nothing),
		model = model |> cpu
		)

	return training_info, [(x -> generalized_anomaly_score_gpu(model|>cpu, x, R=r, L=l, lambda=lam), 
		merge(parameters, (R=r, L=l, test_lambda=lam,)))
		for r in ["mae", "mse"] for l in ["mae", "mse"] for lam = 0.1:0.1:0.9 ]
end

##################
# this is common #
##################
include("run_loop.jl")