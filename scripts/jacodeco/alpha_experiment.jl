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
df = dfs[1]
all_data = load(joinpath(datapath, df))[:jacodata]

iseed = 2
data = all_data[iseed];
ac = data[:ac]
seed = data[:seed]

# this now creates the baseline
val_scores, tst_scores, val_y, tst_y, val_ljd, tst_ljd = 
	data[:val_scores], data[:tst_scores], data[:val_y], data[:tst_y], data[:tst_ljd], data[:val_ljd]




val_scores, tst_scores, val_y, tst_y = get_basic_scores(model_id, ac, ps)
val_auc, tst_auc, alpha = fit_predict_lrnormal(val_scores, tst_scores, val_y, tst_y, 1, p, p_normal)
_val_scores, _val_y, _ = _subsample_data(p, p_normal, val_y, val_scores; seed=seed)
_tst_scores, _tst_y, _ = _subsample_data(p, p_normal, tst_y, tst_scores; seed=seed)

# now load the normal data, subsample them and get their jacodeco for them
orig_data = GenerativeAD.Datasets.load_data(dataset; seed=1, method="leave-one-in", anomaly_class_ind=ac);
orig_data = GenerativeAD.Datasets.normalize_data(orig_data);
val_X, val_Y = orig_data[2];
tst_X, tst_Y = orig_data[3];
_val_X, _val_Y, _ = _subsample_data(p, p_normal, val_Y, val_X; seed=seed);
_tst_X, _tst_Y, _ = _subsample_data(p, p_normal, tst_Y, tst_X; seed=seed);

