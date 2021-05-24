include("utils.jl") # contains most dependencies and the saving function
using GenerativeAD.Models: anomaly_score
using StatsBase: fit!, predict, sample

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
modelname = "Conv-GANomaly"

"""
This returns encodings, parameters and scoring functions in order to reconstruct the experiment. 
This is a slightly updated version of the original run script.
"""
function evaluate(model_data, data, parameters)
	# load the model file, extract params and model
	model = model_data["model"] |> gpu
	
	# compute encodings
	encodings = map(x->cpu(GenerativeAD.Models.encode_mean_gpu(model, x, 32)), (data[1][1], data[2][1], data[3][1]))

	# construct return information - put e.g. the model structure here for generative models
	training_info = (
		fit_t = get(model_data, "fit_t", nothing),
		history = get(model_data, "history", nothing),
		npars = get(model_data, "npars", nothing),
		iters = get(model_data, "iters", nothing),
		model = model |> cpu
		)

	return training_info, [(x -> GenerativeAD.Models.anomaly_score_gpu(generator|>cpu, x; dims=3)[:], parameters)]
end

##################
# this is common #
##################
include("run_loop.jl")