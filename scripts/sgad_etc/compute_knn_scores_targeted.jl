# this only computes knn scores for the best models saved in datadir("sgad_alpha_evaluation_kp/best_models_orig_$(datatype).bson")
# you need to run the basic evaluation scripts nad select_sgvaegan_models_for_probreg.jl 
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
dataset_bm = (datatype == "mvtec") ? "MVTec-AD_$(dataset)" : dataset

bestf = datadir("sgad_alpha_evaluation_kp/best_models_orig_$(datatype).bson")
best_models = load(bestf)

for i in 1:100 # run this over and over until the job time limit is exhausted
    for seed in 1:max_seed
        # outputs
        in_dir = datadir("sgad_encodings/images_$(datatype)/$(modelname)/$(dataset)/ac=$(ac)/seed=$(seed)")

        # outputs
        out_dir = datadir("sgad_latent_scores/images_$(datatype)/$(modelname)/$(dataset)/ac=$(ac)/seed=$(seed)")
        mkpath(out_dir)

        # model dir
        inds = (best_models[:anomaly_class] .== anomaly_class) .& (best_models[:seed] .== seed) .& 
            (best_models[:dataset] .== dataset_bm) .& (best_models[:modelname] .== modelname)
        best_params = best_models[:parameters][inds]

        # from these params extract the correct model_ids and lfs
        parsed_params = map(x->parse_savename("s_$x")[2], best_params)
        best_model_ids = unique([x["init_seed"] for x in parsed_params])
        
        for model_id in best_model_ids
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