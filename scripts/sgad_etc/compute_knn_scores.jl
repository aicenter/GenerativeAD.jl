using DrWatson
@quickactivate
using GenerativeAD
using PyCall
using BSON, FileIO, DataFrames
using ArgParse, StatsBase
import StatsBase: fit!, predict

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

function get_scores(model, x)
    scores = try 
        predict(model, x)
    catch e
        if isa(e, ArgumentError) # this happens in the case when k > number of points
            return NaN # or nothing?
        else
            rethrow(e)
        end
    end
    return scores
end

function fit_predict(k, v, tr_x, val_x, tst_x)
    model = GenerativeAD.Models.knn_constructor(;v=v, k=k)
    fit!(model, Array(transpose(tr_x)))
    val_scores = get_scores(model, Array(transpose(val_x)))
    tst_scores = get_scores(model, Array(transpose(tst_x)))
    return val_scores, tst_scores
end

function compute_knn_score(model_id, in_dir, k, v, out_dir, seed, ac, dataset, modelname; force=false)
    # setup the out file
    outf = joinpath(out_dir, "model_id=$(model_id)_k=$(k)_v=$(v)_score=knn.bson")
    if isfile(outf) && !force
        @info "Skipping computation of $outf as it already exists."
        return
    end

    # load the encodings for this model
    in_f = filter(x->occursin("$(model_id)", x), readdir(in_dir))[1]
    in_d = load(joinpath(in_dir, in_f))

    # now do a knn fit and predict
    data = (
        (in_d[:tr_encodings_shape], in_d[:val_encodings_shape], in_d[:tst_encodings_shape]),
        (in_d[:tr_encodings_background], in_d[:val_encodings_background], in_d[:tst_encodings_background]),
        (in_d[:tr_encodings_foreground], in_d[:val_encodings_foreground], in_d[:tst_encodings_foreground])
        );
    scores = map(x->fit_predict(k,v,x...), data);
    scores = map(i->Array(transpose(hcat([x[i] for x in scores]...))), 1:2)

    # and save them
    output = Dict(
        :parameters => merge(in_d[:parameters], (k=k, v=v)),
        :latent_score_type => "knn",
        :modelname => modelname,
        :dataset => dataset,
        :anomaly_class => ac,
        :seed => seed,
        :tr_scores => NaN,
        :val_scores => scores[1],
        :tst_scores => scores[2],
        :tr_labels => in_d[:tr_labels],
        :val_labels => in_d[:val_labels],
        :tst_labels => in_d[:tst_labels],
        :tr_eval_t => NaN,
        :val_eval_t => NaN,
        :tst_eval_t => NaN    
        )
    save(joinpath(out_dir, outf), output)
    @info "Results writen to $outf."
    output
end

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
            for k in 1:100:2001
                for v in [:delta, :kappa, :gamma]
                    compute_knn_score(model_id, in_dir, k, v, out_dir, seed, ac, dataset, modelname; force=false)
                end
            end
        end
    end
end
