include("utils.jl") # contains most dependencies and the saving function

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
modelname = "aae"
batch_score(scoref, model, x, batchsize=128) =
	vcat(map(y->cpu(scoref(model, gpu(Array(y)))), Flux.Data.DataLoader(x, batchsize=batchsize))...)
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
		model = model |> cpu,
		tr_encodings = encodings[1],
		val_encodings = encodings[2],
		tst_encodings = encodings[3]
		)

	# now return the different scoring functions
	training_info, [
		(x -> batch_score(GenerativeAD.Models.reconstruction_score, model, x), merge(parameters, (score = "reconstruction",))),
		(x -> batch_score(GenerativeAD.Models.reconstruction_score_mean, model, x), merge(parameters, (score = "reconstruction-mean",))),
		]
end

####################################################################
################ THIS PART IS COMMON FOR ALL MODELS ################
include("run_loop.jl")