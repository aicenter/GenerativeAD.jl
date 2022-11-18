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

datapath = datadir("jacodeco/partial_experiment")
dfs = readdir(datapath)


##
df = dfs[2]
all_data = load(joinpath(datapath, df))[:jacodata]

iseed = 2
data = all_data[iseed];
ac = data[:ac]
seed = data[:seed]

# this now creates the baseline
val_scores, tst_scores, val_y, tst_y, val_ljd, tst_ljd = 
	data[:val_scores], data[:tst_scores], data[:val_y], data[:tst_y], data[:tst_ljd], data[:val_ljd]
val_auc_normal, tst_auc_normal, alpha_normal = fit_predict_lrnormal(val_scores, tst_scores, val_y, tst_y)

