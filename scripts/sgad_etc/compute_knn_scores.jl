include("knn_utils.jl")

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
    "--force", "-f"
        action = :store_true
        help = "force recomputing of scores"
end
parsed_args = parse_args(ARGS, s)
@unpack modelname, dataset, datatype, force = parsed_args
max_ac = (datatype == "mvtec") ? 1 : 10
max_seed = (datatype == "mvtec") ? 5 : 1 
ks = (datatype == "mvtec") ? collect(1:4:151) : vcat([1, 31, 61], collect(101:100:2001))

for ac in 1:max_ac
    for seed in 1:max_seed
        # outputs
        in_dir = datadir("sgad_encodings/images_$(datatype)/$(modelname)/$(dataset)/ac=$(ac)/seed=$(seed)")

        # outputs
        out_dir = datadir("sgad_latent_scores/images_$(datatype)/$(modelname)/$(dataset)/ac=$(ac)/seed=$(seed)")
        mkpath(out_dir)

        # model dir
        model_ids = map(x-> Meta.parse(split(split(x, "=")[2], ".")[1]), readdir(in_dir))

        for model_id in model_ids
            for k in ks
                for v in [:delta, :kappa, :gamma]
                    compute_knn_score(model_id, in_dir, k, v, out_dir, seed, ac, dataset, modelname; force=false)
                end
            end
        end
    end
end
