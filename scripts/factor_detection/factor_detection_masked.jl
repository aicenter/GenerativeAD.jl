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

function masked_prediction(model, model_id, outdir, ac, dataset)
	# load the data
	mf_X, mf_Y = GenerativeAD.Datasets.load_wildlife_mnist_raw("test")[2];

	outf = joinpath(outdir, "model_id=$(model_id).bson")

	# do the ranked experiment
	results = map(af->get_prediction_masked(model, ac, af, mf_X, mf_Y), 2:3)

	outdf = Dict(
		:model_id => model_id,
		:y_true_background => results[1][1],
		:y_true_foreground => results[2][1],
		:y_pred_background => results[1][2],
		:y_pred_foreground => results[2][2],
		:bscores_background => results[1][3],
		:fscores_backround => results[2][3],
		:bscores_foreground => results[1][4],
		:fscores_foreground => results[2][4],
		:acc_background => results[1][5],
		:acc_foreground => results[2][5],
		:mean_acc => mean([x[5] for x in results]),
		:method => "masked",
		:dataset => dataset,
		:anomaly_class => ac
		)

	save(outf, :df => outdf)
	@info "saved $outf"
	outdf
end

res = masked_prediction(model, model_id, outdir, ac, dataset)
