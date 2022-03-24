# this script collects the 
using DrWatson
@quickactivate
using GenerativeAD
using PyCall
using FileIO, BSON, DataFrames

n_left = 20
datatype = "leave-one-in"
method = "leave-one-in"
res_df = load(datadir("evaluation/images_leave-one-in_eval.bson"))[:df]

model = "sgvae"
for dataset in ["wildlife_MNIST", "CIFAR10", "SVHN2"]
    for ac in 1:10
        for seed in 1:1
            modelpath = datadir("sgad_models/images_$(datatype)/$(model)/$(dataset)/ac=$(ac)/seed=$(seed)")
            subdf = filter(r->r.dataset == dataset && r.anomaly_class == ac && 
                r.seed == seed && r.modelname == model, res_df)
            subdf = sort(subdf, :val_auc)
            subdf = subdf[1:end-n_left,:]

            for r in eachrow(subdf)
                model_id = parse_savename(r.parameters)[2]["init_seed"]
                dd = joinpath(modelpath, "model_id=$(model_id)")
                if isdir(dd)
                    rm(dd, recursive=true)
                    @info "deleted $dd"
                end
            end
        end
    end
end






model = "vae"
