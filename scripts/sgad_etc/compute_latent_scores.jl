using DrWatson
@quickactivate
using GenerativeAD
using PyCall
using BSON, FileIO, DataFrames
using ArgParse

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
    "datatype"
        default = "leave-one-in"
        arg_type = String
        help = "leave-one-in or mvtec"
    "latent_score_type"
        arg_type = String
        help = "normal, kld or normal_logpx"
        default = "normal"
    "device"
        arg_type = String
        help = "cpu or cuda"
        default = "cpu"
    "--force", "-f"
        action = :store_true
        help = "force recomputing of scores"
end
parsed_args = parse_args(ARGS, s)
@unpack modelname, dataset, datatype, latent_score_type, device, force = parsed_args
max_ac = (datatype == "mvtec") ? 1 : 10
max_seed = (datatype == "mvtec") ? 5 : 1 

function load_sgvae_model(dir, device)
    py"""
import sgad
from sgad.sgvae import SGVAE
from sgad.utils import load_model

def model(dir, device):
    return load_model(SGVAE, dir, device=device)
    """

    return py"model"(dir, device)
end

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
    (tr_X, tr_y), (val_X, val_y), (tst_X, tst_y) = data
    results = map(x->get_latent_scores(model, x), (tr_X, val_X, tst_X));
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
        :tr_labels => tr_y,
        :val_labels => val_y,
        :tst_labels => tst_y,
        :val_scores => latent_scores[2],
        :tst_scores => latent_scores[3],
        :tr_eval_t => ts[1],
        :val_eval_t => ts[2],
        :tst_eval_t => ts[3],    
        )
    outf = joinpath(out_dir, "model_id=$(model_id)_score=$(latent_score_type).bson")
    save(joinpath(out_dir, outf), output)
    @info "Results writen to $outf."
end

           
for ac in 1:max_ac
    for seed in 1:max_seed
        # load the appropriate data
        if datatype == "mvtec"
            data = GenerativeAD.load_data("MVTec-AD", seed=seed, category=dataset, img_size=128)
        else
            data = GenerativeAD.load_data(dataset, seed=seed, anomaly_class_ind=ac, method=datatype);
        end
        data = GenerativeAD.Datasets.normalize_data(data);

        (tr_x, tr_y), (val_x, val_y), (tst_x, tst_y) = data;
        tr_X = Array(permutedims(tr_x, [4,3,2,1]));
        val_X = Array(permutedims(val_x, [4,3,2,1]));
        tst_X = Array(permutedims(tst_x, [4,3,2,1]));
        data = (tr_X, tr_y), (val_X, val_y), (tst_X, tst_y);

        # outputs
        out_dir = datadir("sgad_latent_scores/images_$(datatype)/$(modelname)/$(dataset)/ac=$(ac)/seed=$(seed)")
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
end