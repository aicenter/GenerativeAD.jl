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
seed = 1
method = "leave-one-in"
contamination = 0.0

main_inpath = datadir("experiments/images_leave-one-in_backup/$(modelname)/$(dataset)")
main_savepath = datadir("experiments/images_leave-one-in/$(modelname)/$(dataset)")
mkpath(main_savepath)

# this loop unfortunately cannot be in a function, since loading of bson is only safe ot top level
for ac in 1:10
	data = GenerativeAD.load_data(dataset, seed=seed, anomaly_class_ind=ac, method=method, 
	contamination=contamination)

	inpath = joinpath(main_inpath, "ac=$ac/seed=$seed")
	savepath = joinpath(main_savepath, "ac=$ac/seed=$seed")
	mkpath(savepath)
	fs = readdir(inpath, join=true)
	sfs = filter(x->!(occursin("model", x)), fs)
	mfs = filter(x->(occursin("model", x)), fs)

	@info "Loaded $(length(mfs)) modelfiles in $inpath, processing..."
	for mf in mfs
		# load the bson file on top level, otherwise we get world age problems
		model_data = load(mf)
		parameters = model_data["parameters"]
		training_info, results = evaluate(model_data, data, parameters) # this produces parameters, encodings, score funs
		save_results(parameters, training_info, results, savepath, data, 
			ac, modelname, seed, dataset, contamination) # this computes and saves score and model files
	end
end
