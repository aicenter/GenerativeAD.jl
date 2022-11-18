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

df = dfs[1]
all_data = load(joinpath(datapath, df))[:jacodata]

iseed = 1
data = all_data[iseed];

# this now creates the baseline
val_scores, tst_scores, val_y, tst_y = get_basic_scores(data[:model_id], data[:ac], data[:ps])
