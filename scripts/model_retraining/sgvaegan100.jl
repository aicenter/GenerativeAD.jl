using DrWatson
@quickactivate
using ArgParse
using GenerativeAD
import StatsBase: fit!, predict
using StatsBase
using BSON
using Flux
using JSON

s = ArgParseSettings()
@add_arg_table! s begin
   "config_file"
        arg_type = String
        help = "config"
end
parsed_args = parse_args(ARGS, s)
@unpack config_file = parsed_args
method = "leave-one-in"
contamination = 0.0
cont_string = (contamination == 0.0) ? "" : "_contamination-$contamination"
max_seed = 1
# load the config from json
config = JSON.parsefile(config_file)
ac = config["anomaly_class"]
dataset = config["dataset"]
seed = 1

#######################################################################################
################ THIS PART IS TO BE PROVIDED FOR EACH MODEL SEPARATELY ################
modelname = "sgvaegan100_retrained"
version = 0.4

# sample parameters, should return a Dict of model kwargs 
"""
    sample_params()

Should return a named tuple that contains a sample of model parameters.
"""
function sample_params()
    # convert it to a named tuple
    NamedTuple{Tuple(Symbol.(keys(config)))}(values(config))
end
function GenerativeAD.edit_params(data, parameters)
    idim = size(data[1][1])
    parameters = merge(parameters, (img_dim=idim[1],))
    parameters
end
"""
    fit(data, parameters)

This is the most important function - returns `training_info` and a tuple or a vector of tuples `(score_fun, final_parameters)`.
`training_info` contains additional information on the training process that should be saved, the same for all anomaly score functions.
Each element of the return vector contains a specific anomaly score function - there can be multiple for each trained model.
Final parameters is a named tuple of names and parameter values that are used for creation of the savefile name.
"""
function fit(data, parameters, save_parameters, ac, seed)
    # construct model - constructor should only accept kwargs
    model = GenerativeAD.Models.SGVAEGAN(; parameters...)

    # save intermediate results here
    res_save_path = datadir("sgad_models/images_$(method)$cont_string/$(modelname)/$(dataset)/ac=$(ac)/seed=$(seed)/model_id=$(parameters.init_seed)")
    mkpath(res_save_path)

    # fit train data
    n_epochs = config["nepochs"]
    epoch_iters = ceil(Int, length(data[1][2])/parameters.batch_size)
    save_iter = 2000
    try
         global info, fit_t, _, _, _ = @timed fit!(model, data[1][1]; 
            max_train_time=20*3600/max_seed, workers=4, 
            n_epochs = n_epochs, save_iter = save_iter, save_weights = false, save_path = res_save_path)
    catch e
        # return an empty array if fit fails so nothing is computed
        @info "Failed training due to \n$e"
        return (fit_t = NaN, history=nothing, npars=nothing, model=nothing), [] 
    end
    
    # construct return information - put e.g. the model structure here for generative models
    model = GenerativeAD.Models.SGVAEGAN(info.best_model)
    model.model.eval()
    model.model.move_to("cuda")
    training_info = (
        fit_t = fit_t,
        history = info.history,
        npars = info.npars,
        model = model,
        best_score_type = model.model.best_score_type,
        tr_encodings = nothing,
        val_encodings = nothing,
        tst_encodings = nothing
        )

    # save the final model
    max_iters = length(info.history["iter"])
    mkpath(joinpath(res_save_path, "weights"))
    training_info.model.model.save_weights(
        joinpath(res_save_path, "weights", "$(max_iters).pth")
        )

    # now return the different scoring functions
    parameters = 
    training_info, [
        (x-> predict(model, x, score_type="discriminator", workers=4), merge(save_parameters, (score = "discriminator",))),
        (x-> predict(model, x, score_type="feature_matching", n=10, workers=4), merge(save_parameters, (score = "feature_matching",))),
        (x-> predict(model, x, score_type="reconstruction", n=10, workers=4), merge(save_parameters, (score = "reconstruction",))),
        ]
end

function dropnames(namedtuple::NamedTuple, names::Tuple{Vararg{Symbol}}) 
    keepnames = Base.diff_names(Base._nt_names(namedtuple), names)
    return NamedTuple{keepnames}(namedtuple)
end

####################################################################
################ THIS PART IS COMMON FOR ALL MODELS ################
# only execute this if run directly - so it can be included in other files
savepath = datadir("experiments/images_$(method)$cont_string/$(modelname)/$(dataset)/ac=$(ac)/seed=$(seed)")
mkpath(savepath)

# get data
data = GenerativeAD.load_data(dataset, seed=seed, anomaly_class_ind=ac, method=method, contamination=contamination)
data = GenerativeAD.Datasets.normalize_data(data)

# edit parameters
parameters = sample_params()
edited_parameters = GenerativeAD.edit_params(data, parameters)

@info "Trying to fit $modelname on $dataset with parameters $(edited_parameters)..."
@info "Train/validation/test splits: $(size(data[1][1], 4)) | $(size(data[2][1], 4)) | $(size(data[3][1], 4))"
@info "Number of features: $(size(data[1][1])[1:3])"

# check if a combination of parameters and seed alread exists
if GenerativeAD.check_params(savepath, edited_parameters)
    # fit
    # these parameters will be used in teh savename
    save_parameters = merge(edited_parameters, (version=version,))
    save_parameters = dropnames(save_parameters, (
        :log_var_x_estimate_top, 
        :latent_structure,
        :fixed_mask_epochs,
        :batch_norm,
        :init_type,
        :tau_mask,
        :dataset,
        :anomaly_class
        ))
    training_info, results = fit(data, edited_parameters, save_parameters, ac, seed)

    # save the model separately         
    if training_info.model !== nothing
        tagsave(joinpath(savepath, savename("model", save_parameters, "bson", digits=5)), 
            Dict("fit_t"=>training_info.fit_t,
                 "history"=>training_info.history,
                 "parameters"=>edited_parameters,
                 "tr_encodings"=>training_info.tr_encodings,
                 "val_encodings"=>training_info.val_encodings,
                 "tst_encodings"=>training_info.tst_encodings,
                 "version"=>version,
                 "best_score_type"=>training_info.best_score_type
                 ), 
            safe = true)
        training_info = merge(training_info, 
            (model=nothing,tr_encodings=nothing,val_encodings=nothing,tst_encodings=nothing))
    end

    # here define what additional info should be saved together with parameters, scores, labels and predict times
    save_entries = merge(training_info, (modelname = modelname, seed = seed, 
        dataset = dataset, anomaly_class = ac,
        contamination=contamination))

    # now loop over all anomaly score funs
    for result in results
        GenerativeAD.experiment(result..., data, savepath; save_entries...)
    end
else
    @info "Model already present, try other hyperparameters..."
end
