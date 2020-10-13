using DrWatson
@quickactivate
using ArgParse
using GenerativeAD
using GenerativeAD.Models: anomaly_score, generalized_anomaly_score_gpu
using BSON
using StatsBase: fit!, predict, sample

using Flux
using MLDataPattern


savepath = datadir("experiments/images/Conv-GANomaly")

function models_and_params(path_to_model)
    directories = []
    models = []
    
    for (root, dirs, files) in walkdir(path_to_model)
        for file in files
            push!(directories, root)
        end
    end
    
    for dir in unique(directories)
        fs = filter(x->!(occursin("_#", x)), readdir(dir))
        fs = filter(x->(startswith(x, "model")), fs)
        par = map(x -> DrWatson.parse_savename("_"*x)[2], fs)
        !isempty(fs) ? push!(models, (dir, fs, compare.(par))) : nothing
    end
    return models
end

function compare(params)
    parameters = NamedTuple()
    for (k,v) in zip(collect(keys(params)), collect(values(params)))
        key = (k=="model_batch_size") ? "batch_size" : k
        v = (k=="lr" && v==0) ? 1e-4 : v
        parameters = merge(parameters, NamedTuple{(Symbol(key),)}(v))
    end
    return parameters
end

# code

savepath = datadir("experiments/images/Conv-GANomaly")

models = models_and_params(savepath)
for model in models
    (r, names, params) = model
    for (n,p) in zip(names, params)
        # GANomaly model = (generator, discriminator)
        loaded_mod = BSON.load(joinpath(r,n))["model"]
        
        # get training info 
        spl = split(n, "lr=0_")
        tr_info_name = (length(spl)==2) ? spl[1][7:end]*"lr=0.0001_"*spl[2] : n[7:end]
        
        #print(keys(BSON.load(joinpath(r, tr_info_name))))
        
        training_info_old = BSON.load(joinpath(r, tr_info_name))

        s = split(r, "/") #linux "/" vs windows "\\"
        dataset = String(s[end-2])
        seed = parse(Int, split(s[end-1], "=")[2])
        ac = parse(Int, split(ss[end], "=")[2])
        
        data = GenerativeAD.load_data(dataset, seed=seed, anomaly_class_ind=ac)
        data = GenerativeAD.Models.preprocess_images(data, p)
        
        save_entries = merge(
            (
                fit_t=training_info_old[:fit_t],
                history=training_info_old[:history],
                model = nothing,
                npars = training_info_old[:npars],
                iters = training_info_old[:iters],
            ), 
            (
                modelname = "Conv-GANomaly", 
                seed = seed, 
                dataset = dataset, 
                anomaly_class = ac,
            )
        )
        p = merge(p, (score="testmode",))
        results = [(x -> GenerativeAD.Models.anomaly_score_gpu(loaded_mod[1]|>cpu, x; dims=3)[:], p)]
        for result in results
            GenerativeAD.experiment(result..., data, r; save_entries...)
        end
        @info "model $(n) recomputed and saved"
    end
end