# here we do a small scale experiment that compares our composed anomaly score with the one using J(x)
using DrWatson
@quickactivate
using GenerativeAD
using PyCall
using BSON, FileIO, DataFrames
using EvalMetrics
using OrderedCollections
using Suppressor
using StatsBase
using Random
include("../pyutils.jl")
include("../evaluation/utils/ranks.jl")
include("../evaluation/utils/utils.jl")
include("../supervised_comparison/utils.jl")
include("utils.jl")
base_modelname = "sgvaegan100"
dataset = "SVHN2"
datatype = "leave-one-in"
base_beta = 10.0
scale = true

datapath = datadir("jacodeco/partial_experiment")
dfs = readdir(datapath)


##
df = dfs[10]
all_data = load(joinpath(datapath, df))[:jacodata]

iseed = 1
data = all_data[iseed];

# this now creates the baseline
function get_results(data)
	val_scores, tst_scores, val_y, tst_y, val_ljd, tst_ljd = 
		data[:val_scores], data[:tst_scores], data[:val_y], data[:tst_y], data[:tst_ljd], data[:val_ljd]	
	val_auc_normal, tst_auc_normal, alpha_normal = fit_predict_lrnormal(val_scores, tst_scores, val_y, tst_y)

	if all(isinf.(val_ljd)) || all(isinf.(tst_ljd))
		return 	val_auc_normal, tst_auc_normal, alpha_normal, NaN, NaN, NaN
	end

	val_scores_ljd = hcat(reshape(val_ljd,:,1), val_scores)
	tst_scores_ljd = hcat(reshape(tst_ljd,:,1), tst_scores)
	val_auc_ljd, tst_auc_ljd, alpha_ljd = fit_predict_lrnormal(val_scores_ljd, tst_scores_ljd, val_y, tst_y)

	val_auc_normal, tst_auc_normal, alpha_normal, val_auc_ljd, tst_auc_ljd, alpha_ljd
end

outf = datadir("jacodeco/partial_experiment.bson")
all_results = []
for df in dfs
	all_data = load(joinpath(datapath, df))[:jacodata]
	ac = all_data[1][:ac]
	model_id = all_data[1][:model_id]
	results = map(get_results, all_data)
	res_dict = Dict(
		:val_auc_normal => [x[1] for x in results],
		:tst_auc_normal => [x[2] for x in results],
		:alpha_normal => [x[3] for x in results],
		:val_auc_ljd => [x[4] for x in results],
		:tst_auc_ljd => [x[5] for x in results],
		:alpha_ljd => [x[6] for x in results],
		:ac => ac,
		:model_id => model_id
		)
	push!(all_results, res_dict)
end
save(outf, :results => all_results)
save(datadir("../notebooks_paper/julia_data/jacodeco_partial_experiment.bson"), :results => all_results)
all_results = load(outf)[:results]
