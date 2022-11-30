include("../supervised_comparison/utils.jl")
include("./utils.jl")

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

# just get the data
ldir = datadir("experiments_multifactor/latent_scores/$modelname/$dataset/ac=$ac/seed=1")
lfs = readdir(ldir)
model_ids = map(x->Meta.parse(replace(split(x,"_")[2], "id="=>"")), lfs)
outdir = datadir("factor_identification/prediction_ranked/$modelname/$dataset/ac=$ac/seed=1")

# filter it
modelinds = model_ids .== model_id
final_model_ids = model_ids[modelinds]
final_lfs = lfs[modelinds]

for (lf, model_id) in zip(final_lfs,  final_model_ids)
	ranked_prediction(lf, model_id, outdir, ac, dataset)
end
