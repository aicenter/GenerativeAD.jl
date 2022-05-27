# this is meant to recompute the scores using the latent scores and alpha coefficients for a model
using DrWatson
@quickactivate
using GenerativeAD
using PyCall
using BSON, FileIO, DataFrames
using EvalMetrics
using OrderedCollections
using ArgParse
using Suppressor
using StatsBase
using Random
using GenerativeAD.Evaluation: _prefix_symbol, _get_anomaly_class, _subsample_data
using GenerativeAD.Evaluation: BASE_METRICS, AUC_METRICS
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
    "latent_score_type"
        arg_type = String
        help = "normal, kld, knn or normal_logpx"
        default = "normal"
    "anomaly_class"
        default = nothing
        help = "anomaly class"
    "method"
        default = "logreg"
        help = "logreg or probreg or robreg"
    "--mf_normal"
        action = :store_true
        help = "dont use the original normal data but all the multifactor data not used as anomalies"
    "--anomaly_factors"
        arg_type = Int
        nargs = '+'
        help = "set one or more anomalous factors"
    "--force", "-f"
        action = :store_true
        help = "force recomputing of scores"
end
parsed_args = parse_args(ARGS, s)
@unpack modelname, dataset, latent_score_type, anomaly_class, method, mf_normal, anomaly_factors, force = parsed_args
datatype = "leave-one-in"
acs = isnothing(anomaly_class) ? collect(1:10) : [Meta.parse(anomaly_class)]
seed = 1
nf = length(anomaly_factors)
(nf == 0 || nf > 3) ? error("number of --anomaly_factors must be between 1 and 3") : nothing
save_suffix = mf_normal ? "_mf_normal" : ""

score_type = "logpx"
device = "cpu"
max_seed_perf = 10
scale = true
base_beta = 5.0
init_alpha = [1.0, 0.1, 0.1, 0.1]

# anomaly factors to strings and back
_afs2str(x) = reduce((a,b)->"$(a)$(b)", x)
_str2afs(x) = map(i->Meta.parse(string(x[i])), 1:length(x))
afstring =  _afs2str(anomaly_factors)

function basic_stats(labels, scores)
    try
        roc = EvalMetrics.roccurve(labels, scores)
        auc = EvalMetrics.auc_trapezoidal(roc...)
        prc = EvalMetrics.prcurve(labels, scores)
        auprc = EvalMetrics.auc_trapezoidal(prc...)

        t5 = EvalMetrics.threshold_at_fpr(labels, scores, 0.05)
        cm5 = ConfusionMatrix(labels, scores, t5)
        tpr5 = EvalMetrics.true_positive_rate(cm5)
        f5 = EvalMetrics.f1_score(cm5)

        return auc, auprc, tpr5, f5
    catch e
        if isa(e, ArgumentError)
            return NaN, NaN, NaN, NaN
        else
            rethrow(e)
        end
    end
end

auc_val(labels, scores) = EvalMetrics.auc_trapezoidal(EvalMetrics.roccurve(labels, scores)...)

function perf_at_p_new(p, p_normal, val_scores, val_y, tst_scores, tst_y, init_alpha, base_beta; 
    seed=nothing, scale=true, kwargs...)
    scores, labels, _ = try
        _subsample_data(p, p_normal, val_y, val_scores; seed=seed)
    catch e
        return NaN, NaN
    end
    # if there are no positive samples return NaNs
    if sum(labels) == 0
        val_auc = NaN
        tst_auc = auc_val(tst_y, tst_scores[:,1])
    # if top samples are only positive
    # we cannot train alphas
    # therefore we return the default val performance
    elseif sum(labels) == length(labels) 
        val_auc = NaN
        tst_auc = auc_val(tst_y, tst_scores[:,1])
    # if they are not only positive, then we train alphas and use them to compute 
    # new scores - auc vals on the partial validation and full test dataset
    else
        try
            # get the ligistic regression model
            model = if method == "logreg"
                LogReg()
            elseif method == "probreg"
                ProbReg()
            elseif method == "robreg"
                RobReg(alpha=init_alpha, beta=base_beta/sum(labels))
            else
                error("unknown method $method")
            end
            
            # fit
            if method == "logreg"
                fit!(model, scores, labels)
            elseif method == "probreg"
                fit!(model, scores, labels; verb=false, early_stopping=true, patience=10, balanced=true)
            elseif method == "robreg"
                fit!(model, score, labels; verb=false, early_stopping=true, scale=scale, patience=10,
                    balanced=true)
            end

            #predict
            val_probs = predict(model, scores, scale=scale)
            tst_probs = predict(model, tst_scores, scale=scale)
            val_auc = auc_val(labels, val_probs)
            tst_auc = auc_val(tst_y, tst_probs)
        catch e
            if isa(e, LoadError) || isa(e, ArgumentError)
                val_prec = NaN
                val_auc = NaN
                tst_auc = auc_val(tst_y, tst_scores[:,1])
            else
                rethrow(e)
            end
        end
    end
    return val_auc, tst_auc
end    

nanmean(x) = mean(x[.!isnan.(x)])

function perf_at_p_agg(args...; kwargs...)
    results = [perf_at_p_new(args...; seed=seed, kwargs...) for seed in 1:max_seed_perf]
    return nanmean([x[1] for x in results]), nanmean([x[2] for x in results])
end

function experiment(model_id, lf, ac, latent_dir, save_dir, res_dir, rfs)
    outf = joinpath(save_dir, split(lf, ".")[1] * "_method=$(method).bson")
    @info "$outf"
    if !force && isfile(outf)
        return
    end    

    # load the saved scores
    ldata = load(joinpath(latent_dir, lf))
    rf = filter(x->occursin("$(model_id)", x), rfs)
    if length(rf) < 1
        @info "Something is wrong, original score file for $lf not found"
        return
    end
    rf = rf[1]
    rdata = load(joinpath(res_dir, rf))

    # prepare the data
    if isnan(ldata[:val_scores][1])
        return 
    end
    val_scores_orig = cat(rdata[:val_scores], transpose(ldata[:val_scores]), dims=2);
    tst_scores_orig = cat(rdata[:tst_scores], transpose(ldata[:tst_scores]), dims=2);
    mf_scores = cat(rdata[:mf_scores], transpose(ldata[:mf_scores]), dims=2);
    mf_y = ldata[:mf_labels];

    # split them (pseudo)randomly
    (val_scores, val_y), (tst_scores, tst_y) = GenerativeAD.Datasets.split_multifactor_data(
        anomaly_factors, ac, (transpose(val_scores_orig), transpose(tst_scores_orig)), 
        transpose(mf_scores), mf_y; mf_normal=mf_normal, seed=seed)
    val_scores = Array(transpose(val_scores))
    tst_scores = Array(transpose(tst_scores))

    # setup params
    parameters = merge(ldata[:parameters], (anomaly_factors = afstring, beta=base_beta, 
        init_alpha=init_alpha, scale=scale))
    add_params = split(split(lf, "score")[1], "model_id=$(model_id)")[2]
    param_string = "latent_score_type=$(latent_score_type)_anomaly_factors=$afstring" * add_params
    param_string *= split(rf, ".bson")[1]
    save_modelname = (method == "logreg") ? modelname : modelname*"_$method"

    res_df = @suppress begin
        # prepare the result dataframe
        res_df = OrderedDict()
        res_df["modelname"] = save_modelname
        res_df["dataset"] = dataset
        res_df["phash"] = GenerativeAD.Evaluation.hash(parameters)
        res_df["parameters"] = param_string
        res_df["fit_t"] = NaN
        res_df["tr_eval_t"] = ldata[:tr_eval_t] + rdata[:tr_eval_t]
        res_df["val_eval_t"] = ldata[:val_eval_t] + rdata[:val_eval_t]
        res_df["tst_eval_t"] = ldata[:tst_eval_t] + rdata[:tst_eval_t]
        res_df["seed"] = seed
        res_df["npars"] = rdata[:npars]
        res_df["anomaly_class"] = ac
        res_df["method"] = method
        res_df["score_type"] = score_type
        res_df["latent_score_type"] = latent_score_type
        res_df["anomaly_factors"] = afstring

        # fit the logistic regression - first on all the validation data
        # first, filter out NaNs and Infs
        inds = vec(mapslices(r->!any(r.==Inf), val_scores, dims=2))
        val_scores = val_scores[inds, :]
        val_y = val_y[inds]
        inds = vec(mapslices(r->!any(isnan.(r)), val_scores, dims=2))
        val_scores = val_scores[inds, :]
        val_y = val_y[inds]

        # get the ligistic regression model
        model = if method == "logreg"
            LogReg()
        elseif method == "probreg"
            ProbReg()
        elseif method == "robreg"
            RobReg(alpha=init_alpha, beta=base_beta/sum(val_y))
        else
            error("unknown method $method")
        end
        
        # fit
        if method == "logreg"
            fit!(model, val_scores, val_y)
        elseif method == "probreg"
            fit!(model, val_scores, val_y; verb=false, early_stopping=true, patience=10, balanced=true)
        elseif method == "robreg"
            fit!(model, val_score, val_y; verb=false, early_stopping=true, scale=scale, patience=10,
                balanced=true)
        end
        val_probs = predict(model, val_scores, scale=scale)
        tst_probs = predict(model, tst_scores, scale=scale)
        
        # now fill in the values
        res_df["val_auc"], res_df["val_auprc"], res_df["val_tpr_5"], res_df["val_f1_5"] = 
            basic_stats(val_y, val_probs)
        res_df["tst_auc"], res_df["tst_auprc"], res_df["tst_tpr_5"], res_df["tst_f1_5"] = 
            basic_stats(tst_y, tst_probs)

        # then do the same on a small section of the data
        ps = [100.0, 50.0, 20.0, 10.0, 5.0, 2.0, 1.0, 0.5, 0.2, 0.1]
        auc_ano_100 = [perf_at_p_agg(p/100, 1.0, val_scores, val_y, tst_scores, tst_y, init_alpha, 
            base_beta; scale=scale) for p in ps]
        for (k,v) in zip(map(x->x * "_100", AUC_METRICS), auc_ano_100)
            res_df["val_"*k] = v[1]
            res_df["tst_"*k] = v[2]
        end

        auc_ano_50 = (method == "logreg") ? [perf_at_p_agg(p/100, 0.5, val_scores, val_y, 
                tst_scores, tst_y, init_alpha, base_beta; scale=scale) for p in ps] : repeat([(NaN, NaN)], length(ps))
        for (k,v) in zip(map(x->x * "_50", AUC_METRICS), auc_ano_50)
            res_df["val_"*k] = v[1]
            res_df["tst_"*k] = v[2]
        end

        auc_ano_10 = (method == "logreg") ? [perf_at_p_agg(p/100, 0.1, val_scores, val_y, 
                tst_scores, tst_y, init_alpha, base_beta; scale=scale) for p in ps] : repeat([(NaN, NaN)], length(ps))
        for (k,v) in zip(map(x->x * "_10", AUC_METRICS), auc_ano_10)
            res_df["val_"*k] = v[1]
            res_df["tst_"*k] = v[2]
        end

        prop_ps = [100, 50, 20, 10, 5, 2, 1]
        auc_prop_100 = (method == "logreg") ? [perf_at_p_agg(1.0, p/100, val_scores, val_y, 
            tst_scores, tst_y, init_alpha, base_beta; scale=scale) for p in prop_ps] : repeat([(NaN, NaN)], 7)
        for (k,v) in zip(map(x-> "auc_100_$(x)", prop_ps), auc_prop_100)
            res_df["val_"*k] = v[1]
            res_df["tst_"*k] = v[2]
        end
        res_df
    end
    
    # then save it
    res_df = DataFrame(res_df)
    save(outf, Dict(:df => res_df))
    #@info "Saved $outf."
    res_df
end

for ac in acs
    # we will go over the models that have the latent scores computed - for them we can be sure that 
    # we have all we need
    # we actually don't even need to load the models themselves, just the original (logpx) scores
    # and the latent scores and a logistic regression solver from scikit
    latent_dir = datadir("experiments_multifactor/latent_scores/$(modelname)/$(dataset)/ac=$(ac)/seed=$(seed)")
    lfs = readdir(latent_dir)
    ltypes = map(lf->split(split(lf, "score=")[2], ".")[1], lfs)
    lfs = lfs[ltypes .== latent_score_type]
    model_ids = map(x->Meta.parse(split(split(x, "=")[2], "_")[1]), lfs)

    # make the save dir
    save_dir = datadir("experiments_multifactor/alpha_evaluation$(save_suffix)/$(modelname)/$(dataset)/ac=$(ac)/seed=$(seed)/af=$(afstring)")
    mkpath(save_dir)
    @info "Saving data to $(save_dir)..."

    # top score files
    res_dir = datadir("experiments_multifactor/base_scores/$(modelname)/$(dataset)/ac=$(ac)/seed=$(seed)")
    rfs = readdir(res_dir)
    rfs = filter(x->occursin(score_type, x), rfs)

    for (model_id, lf) in zip(model_ids, lfs)
        experiment(model_id, lf, ac, latent_dir, save_dir, res_dir, rfs)
    end
    @info "Done."
end
