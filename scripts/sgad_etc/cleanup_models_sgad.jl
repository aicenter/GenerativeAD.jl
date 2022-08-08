# this script collects the top models and deltes the rest
using DrWatson
@quickactivate
using GenerativeAD
using PyCall
using FileIO, BSON, DataFrames


### LEAVE ONE IN
datatype = "leave-one-in"
method = "leave-one-in"
res_df = load(datadir("evaluation/images_leave-one-in_eval.bson"))[:df]

n_left = 20
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

model = "cgn"
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

n_left = 10
for model in ["vae", "fmgan", "DeepSVDD", "fAnoGAN"]
    for dataset in ["cocoplaces"]
        for ac in 1:10
            for seed in 1:1
                modelpath = datadir("experiments/images_$(datatype)/$(model)/$(dataset)/ac=$(ac)/seed=$(seed)")
                mfs = filter(x->occursin("model", x), readdir(modelpath))
                subdf = filter(r->r.dataset == dataset && r.anomaly_class == ac && 
                    r.seed == seed && r.modelname == model, res_df)

                subdf[:model_id] = map(r->parse_savename(r.parameters)[2]["init_seed"], eachrow(subdf))
                subdf = subdf[:,[:model_id, :val_auc]]
                subdf = combine(groupby(subdf, :model_id), :val_auc=>maximum=>:val_auc)
                subdf = sort(subdf, :val_auc)[1:end-n_left,:]


                for model_id in subdf.model_id
                    _mfs = filter(x->occursin("$(model_id)", x), mfs)
                    if length(_mfs) > 0
                        mf = joinpath(modelpath, _mfs[1])
                        rm(mf)
                        @info "removed $mf"
                    end
                end
            end
        end
    end
end









### MVTEC
datatype = "mvtec"
method = "mvtec"
res_df = load(datadir("evaluation/images_mvtec_eval.bson"))[:df]

n_left = 20
model = "sgvae"
for category in ["bottle", "capsule", "grid", "metal_nut", "pill", "transistor", "wood"]
    for ac in 1:1
        for seed in 1:5
            modelpath = datadir("sgad_models/images_$(datatype)/$(model)/$(category)/ac=$(ac)/seed=$(seed)")
            subdf = filter(r->r.dataset == "MVTec-AD_$(category)" && 
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

model = "cgn"
for category in ["bottle", "capsule", "grid", "metal_nut", "pill", "transistor", "wood"]
    for ac in 1:1
        for seed in 1:5
            modelpath = datadir("sgad_models/images_$(datatype)/$(model)/$(category)/ac=$(ac)/seed=$(seed)")
            subdf = filter(r->r.dataset == "MVTec-AD_$(category)" && 
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

n_left = 10
for model in ["vae", "fmgan", "DeepSVDD", "fAnoGAN"]
    for category in ["bottle", "capsule", "metal_nut", "pill"]
        for ac in 1:1
            for seed in 1:5
                modelpath = datadir("experiments/images_$(datatype)/$(model)/$(category)/ac=$(ac)/seed=$(seed)")
                mfs = filter(x->occursin("model", x), readdir(modelpath))
                subdf = filter(r->r.dataset == "MVTec-AD_$(category)" && 
                    r.seed == seed && r.modelname == model, res_df)

                subdf[:model_id] = map(r->parse_savename(r.parameters)[2]["init_seed"], eachrow(subdf))
                subdf = subdf[:,[:model_id, :val_auc]]
                subdf = combine(groupby(subdf, :model_id), :val_auc=>maximum=>:val_auc)
                subdf = sort(subdf, :val_auc)[1:end-n_left,:]


                for model_id in subdf.model_id
                    _mfs = filter(x->occursin("$(model_id)", x), mfs)
                    if length(_mfs) > 0
                        mf = joinpath(modelpath, _mfs[1])
                        rm(mf)
                        @info "removed $mf"
                    end
                end
            end
        end
    end
end
