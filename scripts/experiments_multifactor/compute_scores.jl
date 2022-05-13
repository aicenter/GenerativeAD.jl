# this is meant to recompute the scores of models on the factorized datasets such as wildlife MNIST
using DrWatson
@quickactivate
using GenerativeAD
using ArgParse
using DataFrames
using StatsBase
using Dates
#using Flux, PyCall
include("../pyutils.jl")

s = ArgParseSettings()
@add_arg_table! s begin
	"modelname"
		arg_type = String
		help = "model name"
    "dataset"
        default = "wildlife_MNIST"
        arg_type = String
        help = "dataset"
    "device"
    	default = "cpu"
    	arg_type = String
    	help = "cpu or cuda"
    "--force", "-f"
        action = :store_true
        help = "force recomputing of scores"
end
parsed_args = parse_args(ARGS, s)
@unpack modelname, dataset, device, force = parsed_args
method = "leave-one-in"

function multifactor_experiment(score_fun, parameters, data, normal_label, savepath; force=true, verb=true, 
    save_entries...)
    # first create savename and check if it is not already present
    savef = joinpath(savepath, savename(parameters, "bson", digits=5))
    if isfile(savef) && !(force)
        verb ? (@info "$savef already present, skipping") : nothing
        return nothing
    end
    
    # get the data
    tr_data, val_data, tst_data, mf_data = data

    # extract scores
    tr_scores, tr_eval_t, _, _, _ = @timed score_fun(tr_data[1])
    val_scores, val_eval_t, _, _, _ = @timed score_fun(val_data[1])
    tst_scores, tst_eval_t, _, _, _ = @timed score_fun(tst_data[1])
    mf_scores, mf_eval_t, _, _, _ = @timed score_fun(mf_data[1])

    # now save the stuff
    result = (
        parameters = parameters,
        tr_scores = tr_scores,
        tr_eval_t = tr_eval_t,
        val_scores = val_scores,
        val_eval_t = val_eval_t,
        tst_scores = tst_scores,
        tst_eval_t = tst_eval_t,
        mf_scores = mf_scores,
        mf_labels = mf_data[2],
        mf_eval_t = mf_eval_t,
        normal_label = normal_label
        )
    result = Dict{Symbol, Any}([sym=>val for (sym,val) in pairs(merge(result, save_entries))]) # this has to be a Dict 
    tagsave(savef, result, safe = true)
    verb ? (@info "Results saved to $savef") : nothing
    result
end

function compute_scores(mf, model_id, expfs, paths, ac; verb=true)
    # paths
    exppath, outdir = paths

    # load the original experiment file
    expf = filter(x->occursin("$(model_id)", x), expfs)
    expf = filter(x->occursin("model", x), expf)
    # put a return here in case this is empty
    if length(expf) == 0
        return nothing
    end
    expf = joinpath(exppath, expf[1])
    #exptime = mtime(expf)
    expdata = load(expf)

    # this will have to be specific for each modelname
    if modelname == "sgvae"
        model = GenerativeAD.Models.SGVAE(load_sgvae_model(mf, device))
    else
        error("unknown modelname $modelname")
    end

    # setup the parameters to be saved
    save_parameters = dropnames(expdata["parameters"], (
        :log_var_x_estimate_top, 
        :latent_structure
        ))
    save_entries = Dict(
        :anomaly_class => ac,
        :dataset => dataset,
        :modelname => modelname,
        :npars => get(expdata, "npars", nothing),
        :seed => seed
        )
    data = (orig_data[1], orig_data[2], orig_data[3], multifactor_data);
    
    # compose the return function
    results = 
    if modelname == "sgvae"
        [
        (x-> StatsBase.predict(model, x, score_type="logpx", n=10, workers=4), merge(save_parameters, (score = "logpx",))),
        ]
    else
        error("predict functions for $modelname not implemented")
    end

    # now run and save the experiments
    for result in results
        multifactor_experiment(result..., data, ac, outdir; force=force, verb=verb, save_entries...)
    end
end

function dropnames(namedtuple::NamedTuple, names::Tuple{Vararg{Symbol}}) 
    keepnames = Base.diff_names(Base._nt_names(namedtuple), names)
    return NamedTuple{keepnames}(namedtuple)
end

seed = 1
for ac in 1:10
    # load the original train/val/test split
    orig_data = GenerativeAD.load_data(dataset, seed=seed, anomaly_class_ind=ac, method=method);
    # also load the new data for inference
    if dataset == "wildlife_MNIST"
        multifactor_data = GenerativeAD.Datasets.load_wildlife_mnist_raw("test")[2];
    else
        error("unkown dataset $(dataset)")
    end

    # setup paths
    if modelname in ["sgvae", "cgn"]
    	main_modelpath = datadir("sgad_models/images_$(method)/$(modelname)/$(dataset)")
    end

    # save dir
    outdir = datadir("experiments/images_multifactor/$(modelname)/$(dataset)/ac=$(ac)/seed=$(seed)")
    mkpath(outdir)

    # original experiment dir
    exppath = datadir("experiments/images_$(method)/$(modelname)/$(dataset)/ac=$(ac)/seed=$(seed)")
    expfs = readdir(exppath)

    # model dir
    modelpath = joinpath(main_modelpath, "ac=$(ac)/seed=$(seed)")
    mfs = readdir(modelpath, join=true)
    model_ids = map(x->Meta.parse(split(x, "model_id=")[2]), mfs)

    @info "processing $(modelpath)..."
    for (mf, model_id) in zip(mfs, model_ids)
        compute_scores(mf, model_id, expfs, (exppath, outdir), ac; verb=true)
    end 
end