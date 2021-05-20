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

modelname = "fmgan"
batch_score(scoref, model, x, batchsize=512) =
	vcat(map(y->vec(cpu(scoref(model, gpu(Array(y))))), Flux.Data.DataLoader(x, batchsize=batchsize))...)
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
		npars = get(model_data, "npars", nothing),
		model = model |> cpu
		)
	# now return the different scoring functions
	training_info, [
		(x -> 1f0 .- batch_score(GenerativeAD.Models.discriminate, model, x), parameters)
		]
end

####################################################################
################ THIS PART IS COMMON FOR ALL MODELS ################
include("run_loop.jl")