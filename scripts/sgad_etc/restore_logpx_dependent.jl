using DrWatson
@quickactivate
using GenerativeAD
using PyCall
using FileIO, BSON, DataFrames

function predict(model, x)
    model.eval()
    x = Array(permutedims(x, [4,3,2,1]))
    scores = model.predict(x, score_type="logpx", n=1, workers=4)
end

function load_model(dir, device)
    py"""
import sgad
from sgad.sgvae import SGVAE
from sgad.utils import load_model

def model(dir, device):
    return load_model(SGVAE, dir, device=device)
    """

    return py"model"(dir, device)
end

function recompute_scores(f, modelpath, data)
    # get the results file and model file
    init_seed = parse_savename(f)[2]["init_seed"]
    md = joinpath(modelpath, "model_id=$(init_seed)") 

    # now load the results
    results = load(f)

    # load the model
    model = load_model(md, "cpu")

    # compute new scores
    tr_scores = predict(model, data[1][1])
    val_scores = predict(model, data[2][1])
    tst_scores = predict(model, data[3][1])

    # and overwrite them
    results[:tr_scores] = tr_scores
    results[:val_scores] = val_scores
    results[:tst_scores] = tst_scores
    save(f, results)
    @info "recomputed $f"
end

datatype = "leave-one-in"
method = "leave-one-in"

for dataset in ["wildlife_MNIST", "CIFAR10", "SVHN2"]
    for ac in 1:10
        for seed in 1:1
            # load the input data
            data = GenerativeAD.load_data(dataset, seed=seed, anomaly_class_ind=ac, method=method);

            # collect the files to be inspected
            modelpath = datadir("sgad_models/images_$(datatype)/sgvae/$(dataset)/ac=$(ac)/seed=$(seed)")
            inpath = datadir("experiments/images_$(datatype)/sgvae/$(dataset)/ac=$(ac)/seed=$(seed)")
            res_fs = readdir(inpath)
            res_fs = filter(x->!occursin("model", x), res_fs)
            res_fs = filter(x->occursin("mask_dependent", x), res_fs)

            for f in res_fs
                f = joinpath(inpath, f)
                recompute_scores(f, modelpath, data)
            end
        end
    end
end

for category in ["wood"," transistor", "grid", "bottle", "metal_nut", "pill", "capsule"]
    for ac in 1:1
        for seed in 1:5
            # load the input data
            data = GenerativeAD.load_data("MVTec-AD", seed=seed, category=category, img_size=128)

            # collect the files to be inspected
            modelpath = datadir("sgad_models/images_$(datatype)/sgvae/$(category)/ac=$(ac)/seed=$(seed)")
            inpath = datadir("experiments/images_$(datatype)/sgvae/$(category)/ac=$(ac)/seed=$(seed)")
            res_fs = readdir(inpath)
            res_fs = filter(x->!occursin("model", x), res_fs)
            res_fs = filter(x->occursin("mask_dependent", x), res_fs)

            for f in res_fs
                f = joinpath(inpath, f)
                recompute_scores(f, modelpath, data)
            end
        end
    end
end
