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
	in_ch = size(data[1][1],3)
	isize = maximum([size(data[1][1],1),size(data[1][1],2)])
	isize = (isize % 16 != 0) ? isize + 16 - isize % 16 : isize

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
		if haskey(model_data, "parameters")
			parameters = model_data["parameters"]
		else # this is in case parameters are not saved in the model file
			init_seed = DrWatson.parse_savename(mf)[2]["init_seed"]
			sf = sfs[map(x->DrWatson.parse_savename(x)[2]["init_seed"], sfs) .== init_seed][1]
			score_data = load(sf)
			parameters = score_data[:parameters]
		end
		parameters = merge(parameters, (isize=isize, in_ch = in_ch, out_ch = 1))
		# update parameter
		data = GenerativeAD.Models.preprocess_images(data, parameters)

		try
			training_info, results = evaluate(model_data, data, parameters) # this produces parameters, encodings, score funs
			save_results(parameters, training_info, results, savepath, data, 
				ac, modelname, seed, dataset, contamination) # this computes and saves score and model files
		catch e
			if isa(e, LoadError)
				@warn "$mf failed during result evaluation due to $e"
			else
				rethrow(e)
			end
		end
	end
end


