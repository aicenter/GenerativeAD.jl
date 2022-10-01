include("../sgad_etc/knn_utils.jl")

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
    "anomaly_class"
        default = nothing
        help = "set the anomaly class to be computed"
    "--force", "-f"
        action = :store_true
        help = "force recomputing of scores"
end
parsed_args = parse_args(ARGS, s)
@unpack modelname, dataset, anomaly_class, force = parsed_args
datatype = "leave-one-in"
acs = isnothing(anomaly_class) ? collect(1:10) : [Meta.parse(anomaly_class)]
seed = 1
ks = (dataset == "wildlife_MNIST") ? vcat([1, 31, 61], collect(101:100:2001)) : collect(1:10:201)

for ac in acs
    # outputs
    in_dir = datadir("experiments_multifactor/encodings/$(modelname)/$(dataset)/ac=$(ac)/seed=$(seed)")

    # outputs
    out_dir = datadir("experiments_multifactor/latent_scores/$(modelname)/$(dataset)/ac=$(ac)/seed=$(seed)")
    mkpath(out_dir)

    # model dir
    model_ids = map(x-> Meta.parse(split(split(x, "=")[2], ".")[1]), readdir(in_dir))
    
    for k in ks
        for model_id in model_ids
            for v in [:delta, :kappa, :gamma]
                compute_knn_score_multifactor(model_id, in_dir, k, v, out_dir, seed, ac, dataset, modelname; force=false)
            end
        end
    end
end
