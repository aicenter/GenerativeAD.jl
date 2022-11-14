# this is meant to compute the logDetJaco score (part of jacodeco) for the python sgad models
using DrWatson
@quickactivate
using ArgParse
using GenerativeAD
import StatsBase: fit!, predict
using BSON, FileIO
using Random

include("../pyutils.jl")

s = ArgParseSettings()
@add_arg_table! s begin
	"modelname"
		default = "sgvaegan100"
		arg_type = String
		help = "model name"
	"dataset"
		default = "SVHN2"
		arg_type = String
		help = "dataset"
	"anomaly_class"
		default = 1
		arg_type = Int
		help = "which anomaly class to compute"
	"--force"
        action = :store_true
        help = "force recomputing of scores"
end
parsed_args = parse_args(ARGS, s)
@unpack dataset, modelname, anomaly_class, force = parsed_args
method = "leave-one-in"
ac = anomaly_class
data = GenerativeAD.load_data(dataset, seed=1, anomaly_class_ind=ac, method=method)
data = GenerativeAD.Datasets.normalize_data(data)
device = "cuda"
seed = 1

modelpath = datadir("sgad_models/images_$(method)/$(modelname)/$(dataset)/ac=$(ac)/seed=$(seed)")
mdirs = readdir(modelpath, join=true)
outpath = datadir("jacodeco/images_$(method)/$(modelname)/$(dataset)/ac=$(ac)/seed=$(seed)")
mkpath(outpath)

function compute_log_jacodet_scores(md)
	outf = joinpath(outpath, "$(basename(md)).bson")
	if isfile(outf) && !force
		@info "$outf present, skipping"
	end

	model = if occursin("sgvaegan", modelname)
		GenerativeAD.Models.SGVAEGAN(load_sgvaegan_model(md, device))
	else
		throw("unknown model type $modelname")
	end

	score(x) = predict(model, x, score_type="log_jacodet", workers=2)

	val_scores, tst_scores = score(data[2][1]), score(data[3][1])
	outd = Dict(
		:tr_scores => nothing,
		:val_scores => val_scores,
		:tst_scores => tst_scores,
		:model_id => Meta.parse(split(basename(md),"=")[2]), 
		:dataset => dataset,
		:modelname => modelname,
		:anomaly_class => ac,
		:seed => seed
		)
	save(outd, )
	@info "saved $outf"
	return outd
end

for md in shuffle(mdirs)
	compute_log_jacodet_scores(md)
end