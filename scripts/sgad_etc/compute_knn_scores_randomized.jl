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
    "anomaly_class"
        default = 1
        arg_type = Int
        help = "anomaly class"
    "--force", "-f"
        action = :store_true
        help = "force recomputing of scores"
end
parsed_args = parse_args(ARGS, s)
@unpack modelname, dataset, datatype, force, anomaly_class = parsed_args
max_seed = (datatype == "mvtec") ? 5 : 1 
ks = (datatype == "mvtec") ? collect(1:4:151) : vcat([1, 31, 61], collect(101:100:2001))
ks = (dataset == "cocoplaces") ? collect(1:10:201) : ks
ac = anomaly_class
max_tries = 20

while true # run this over and over until the job time limit is exhausted
    for seed in 1:max_seed
        # outputs
        in_dir = datadir("sgad_encodings/images_$(datatype)/$(modelname)/$(dataset)/ac=$(ac)/seed=$(seed)")

        # outputs
        out_dir = datadir("sgad_latent_scores/images_$(datatype)/$(modelname)/$(dataset)/ac=$(ac)/seed=$(seed)")
        mkpath(out_dir)

        # model dir
        model_ids = map(x-> Meta.parse(split(split(x, "=")[2], ".")[1]), readdir(in_dir))
        for model_id in sample(model_ids, length(model_ids), replace=false) 
            res = nothing
            ntries = 1
            while isnothing(res) && ntries <= max_tries # the script might get stuck here
                k = sample(ks, 1)[1]
                v = sample([:delta, :kappa, :gamma], 1)[1]
                res = compute_knn_score(model_id, in_dir, k, v, out_dir, seed, ac, dataset, modelname; 
                    force=force)                    
                ntries += 1
            end
        end
    end
end