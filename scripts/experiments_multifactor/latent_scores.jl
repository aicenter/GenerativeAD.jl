using DrWatson
@quickactivate
using GenerativeAD
using PyCall
using BSON, FileIO, DataFrames
using ArgParse, StatsBase
include("../pyutils.jl")

s = ArgParseSettings()
@add_arg_table! s begin
   "modelname"
        default = "sgvae"
        arg_type = String
        help = "modelname"
    "dataset"
        default = "wildlife_MNIST"
        arg_type = String
        help = "dataset or mvtec category"
    "device"
        arg_type = String
        help = "cpu or cuda"
        default = "cpu"
    "latent_score_type"
        arg_type = String
        help = "normal, kld or normal_logpx"
        default = "normal"
    "anomaly_class"
        default = nothing
        help = "set the anomaly class to be computed"
    "--force", "-f"
        action = :store_true
        help = "force recomputing of scores"
end
parsed_args = parse_args(ARGS, s)
@unpack modelname, dataset, latent_score_type, device, anomaly_class, force = parsed_args
acs = isnothing(anomaly_class) ? collect(1:10) : [Meta.parse(anomaly_class)]
seed = 1
datatype = "leave-one-in"

function get_latent_scores(model, x)
    scores, t, _, _, _ = @timed model.all_scores(x, n=10, score_type="logpx", 
        latent_score_type=latent_score_type, workers=4)
    return  scores[2:end,:], t
end

function compute_save_scores(model_id, model_dir, device, data, res_fs, res_dir, 
    out_dir, latent_score_type, seed, ac, dataset, modelname; force=false)
    # first check whether the scores were not already computed
    outf = joinpath(out_dir, "model_id=$(model_id)_score=$(latent_score_type).bson")
    if isfile(outf) && !force
        @info "Skipping computation of $outf as it already exists."
        return
    end

    # load the model
    md = joinpath(model_dir, "model_id=$(model_id)")
    if !("weights" in readdir(md))
        @info "Model weights not found in $md."
        return
    end    
    model = load_sgvae_model(md, device);
    model.eval();

    # load the original result file for this model
    res_f = filter(x->occursin("$(model_id)", x), res_fs)
    res_f = filter(x->!occursin("model", x), res_f)
    if length(res_f) < 1
        @info "Model result data not found for $(md)."
        return
    end
    res_f = res_f[1]
    res_d = load(joinpath(res_dir, res_f))

    # compute the results
    (tr_X, tr_y), (val_X, val_y), (tst_X, tst_y), (mf_X, mf_y) = data
    tr_X = Array(permutedims(tr_X, [4,3,2,1]));
    val_X = Array(permutedims(val_X, [4,3,2,1]));
    tst_X = Array(permutedims(tst_X, [4,3,2,1]));
    mf_X = Array(permutedims(mf_X, [4,3,2,1]));
    results = try
        map(x->get_latent_scores(model, x), (tr_X, val_X, tst_X, mf_X));
    catch e
        if isa(e, PyCall.PyError)
            @info "Python error during computation of $(res_f)."
            return
        else
            rethrow(e)
            return
        end
    end

    latent_scores = [x[1] for x in results];
    ts = [x[2] for x in results];

    # and save them
    output = Dict(
        :parameters => res_d[:parameters],
        :latent_score_type => latent_score_type,
        :modelname => modelname,
        :dataset => dataset,
        :anomaly_class => ac,
        :seed => seed,
        :tr_scores => latent_scores[1],
        :val_scores => latent_scores[2],
        :tst_scores => latent_scores[3],
        :mf_scores => latent_scores[4],
        :tr_labels => tr_y,
        :val_labels => val_y,
        :tst_labels => tst_y,
        :mf_labels => mf_y,
        :tr_eval_t => ts[1],
        :val_eval_t => ts[2],
        :tst_eval_t => ts[3],    
        )
    save(joinpath(out_dir, outf), output)
    @info "Results writen to $outf."
    output
end

for ac in acs
    # load data
    if dataset == "wildlife_MNIST"
        # load the original train/val/test split
        orig_data = GenerativeAD.load_data(dataset, seed=seed, anomaly_class_ind=ac, method=datatype);
        # and also the multifactor dataset
        multifactor_data = GenerativeAD.Datasets.load_wildlife_mnist_raw("test")[2];
    else
        error("unkown dataset $(dataset)")
    end

    # only use the normal original data
    normal_inds = map(is->is .== 0, (orig_data[1][2], orig_data[2][2], orig_data[3][2]));
    data = (
    (orig_data[1][1][:,:,:,normal_inds[1]], orig_data[1][2][normal_inds[1]]),
    (orig_data[2][1][:,:,:,normal_inds[2]], orig_data[2][2][normal_inds[2]]),
    (orig_data[3][1][:,:,:,normal_inds[3]], orig_data[3][2][normal_inds[3]]),
     multifactor_data);

    # outputs
    out_dir = datadir("experiments_multifactor/latent_scores/$(modelname)/$(dataset)/ac=$(ac)/seed=$(seed)")
    mkpath(out_dir)

    # model dir
    model_dir = datadir("sgad_models/images_$(datatype)/$(modelname)/$(dataset)/ac=$(ac)/seed=$(seed)")
    model_ids = map(x-> Meta.parse(split(x, "=")[2]), readdir(model_dir))
    res_dir = datadir("experiments/images_$(datatype)/$(modelname)/$(dataset)/ac=$(ac)/seed=$(seed)")
    res_fs = readdir(res_dir)

    for model_id in model_ids
        compute_save_scores(model_id, model_dir, device, data, res_fs, res_dir, 
            out_dir, latent_score_type, seed, ac, dataset, modelname, force=force)
    end
end
