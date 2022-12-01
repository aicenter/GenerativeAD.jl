include("../supervised_comparison/utils.jl")
include("./utils.jl")
include("../pyutils.jl")

s = ArgParseSettings()
@add_arg_table! s begin
	"modelname"
        default = "sgvaegan100"
		arg_type = String
		help = "model name"
    "dataset"
        default = "wildlife_MNIST"
        arg_type = String
        help = "dataset"
    "anomaly_class"
    	default = 1
        arg_type = Int
        help = "anomaly class"
    "model_id"
    	default = 32131929
    	arg_type = Int
    	help = "model id"
    "--force", "-f"
        action = :store_true
        help = "force recomputing of scores"
end
parsed_args = parse_args(ARGS, s)
@unpack modelname, dataset, anomaly_class, model_id, force = parsed_args
ac = anomaly_class

# find and load the model
mainpath = datadir("sgad_models/images_leave-one-in/$modelname/$dataset/ac=$ac/seed=1/")
mfs = readdir(mainpath)
modelpath = joinpath(mainpath, filter(x->occursin(string(model_id), x), mfs)[1])
model = GenerativeAD.Models.SGVAEGAN(load_sgvaegan_model(modelpath, "cuda"))
model.model.eval();
outdir = datadir("factor_identification/prediction_masked/$modelname/$dataset/ac=$ac/seed=1")

# compute the predictions
res = map(iexp->masked_prediction(model, model_id, outdir, ac, dataset, iexp), 1:10)
