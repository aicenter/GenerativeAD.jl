using DrWatson
@quickactivate
using ArgParse
using GenerativeAD
import StatsBase: fit!, predict
using StatsBase
using BSON
using Flux

s = ArgParseSettings()
@add_arg_table! s begin
   "max_seed"
        default = 1
        arg_type = Int
        help = "seed"
    "dataset"
        default = "MNIST"
        arg_type = String
        help = "dataset"
    "anomaly_classes"
        arg_type = Int
        default = 10
        help = "number of anomaly classes"
    "method"
        arg_type = String
        default = "leave-one-in"
        help = "method for data creation -> \"leave-one-out\" or \"leave-one-in\" "
    "contamination"
        arg_type = Float64
        help = "contamination rate of training data"
        default = 0.0
end
parsed_args = parse_args(ARGS, s)
@unpack dataset, max_seed, anomaly_classes, method, contamination = parsed_args
cont_string = (contamination == 0.0) ? "" : "_contamination-$contamination"

#######################################################################################
################ THIS PART IS TO BE PROVIDED FOR EACH MODEL SEPARATELY ################
modelname = "sgvaegan"
version = 0.1

# sample parameters, should return a Dict of model kwargs 
"""
    sample_params()

Should return a named tuple that contains a sample of model parameters.
"""
function sample_params()
    weights_texture = (0.01, 0.05, 0.0, 0.01)
    par_vec = (
        2 .^(3:8), 
        2 .^(3:6), 
        map(x->x .* weights_texture, [1, 5, 10, 50, 100, 500, 1000]),
        vcat(10 .^(-1.0:3.0), 0.5 .* 10 .^(-1.0:3.0)),
        vcat(10 .^(-1.0:3.0), 0.5 .* 10 .^(-1.0:3.0)),
        0.1:0.1:0.3,
        0:3,
        2 .^(4:7), 
        ["orthogonal", "normal"], 
        0.01:0.01:0.1, 
        1:Int(1e8), 
        10f0 .^(-4:0.1:-3),
        10f0 .^(-2:1.0:3.0),
        [0, 2, 5, 8],
        )
    argnames = (
        :z_dim, 
        :h_channels,
        :weights_texture, 
        :weight_binary,
        :weight_mask,
        :tau_mask,
        :latent_structure,
        :batch_size, 
        :init_type, 
        :init_gain, 
        :init_seed, 
        :lr,
        :fm_alpha,
        :fm_depth,
        )
    parameters = (;zip(argnames, map(x->sample(x, 1)[1], par_vec))...)
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
    input_range = (-1,1)
    model = GenerativeAD.Models.SGVAEGAN(;input_range=input_range, parameters...)

    # save intermediate results here
    res_save_path = datadir("sgad_models/images_$(method)$cont_string/$(modelname)/$(dataset)/ac=$(ac)/seed=$(seed)/model_id=$(parameters.init_seed)")
    mkpath(res_save_path)

    # fit train data
    n_epochs = 200
    epoch_iters = ceil(Int, length(data[1][2])/parameters.batch_size)
    save_iter = epoch_iters*10
    try
         global info, fit_t, _, _, _ = @timed fit!(model, data[1][1]; 
            max_train_time=20*3600/max_seed/anomaly_classes, workers=4,
            n_epochs = n_epochs, save_iter = save_iter, save_weights = false, save_path = res_save_path)
    catch e
        # return an empty array if fit fails so nothing is computed
        @info "Failed training due to \n$e"
        return (fit_t = NaN, history=nothing, npars=nothing, model=nothing), [] 
    end
    
    # construct return information - put e.g. the model structure here for generative models
    training_info = (
        fit_t = fit_t,
        history = info.history,
        npars = info.npars,
        model = model,
        tr_encodings = nothing,
        val_encodings = nothing,
        tst_encodings = nothing
        )

    # save the final model
    model.model.eval()
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
if abspath(PROGRAM_FILE) == @__FILE__
    # set a maximum for parameter sampling retries
    try_counter = 0
    max_tries = 10*max_seed
    while try_counter < max_tries
        parameters = sample_params()

        for seed in 1:max_seed
            for i in 1:anomaly_classes
                savepath = datadir("experiments/images_$(method)$cont_string/$(modelname)/$(dataset)/ac=$(i)/seed=$(seed)")
                mkpath(savepath)

                # get data
                data = GenerativeAD.load_data(dataset, seed=seed, anomaly_class_ind=i, method=method, contamination=contamination)
                data = GenerativeAD.Datasets.normalize_data(data)

                # edit parameters
                edited_parameters = GenerativeAD.edit_params(data, parameters)

                @info "Trying to fit $modelname on $dataset with parameters $(edited_parameters)..."
                @info "Train/validation/test splits: $(size(data[1][1], 4)) | $(size(data[2][1], 4)) | $(size(data[3][1], 4))"
                @info "Number of features: $(size(data[1][1])[1:3])"

                # check if a combination of parameters and seed alread exists
                if GenerativeAD.check_params(savepath, edited_parameters)
                    # fit
                    save_parameters = merge(edited_parameters, (version=version,))
                    save_parameters = dropnames(save_parameters, (
                        :log_var_x_estimate_top, 
                        :latent_structure
                        ))
                    training_info, results = fit(data, edited_parameters, save_parameters, i, seed)

                    # save the model separately         
                    if training_info.model !== nothing
                        tagsave(joinpath(savepath, savename("model", save_parameters, "bson", digits=5)), 
                            Dict("fit_t"=>training_info.fit_t,
                                 "history"=>training_info.history,
                                 "parameters"=>edited_parameters,
                                 "tr_encodings"=>training_info.tr_encodings,
                                 "val_encodings"=>training_info.val_encodings,
                                 "tst_encodings"=>training_info.tst_encodings,
                                 "version"=>version
                                 ), 
                            safe = true)
                        training_info = merge(training_info, 
                            (model=nothing,tr_encodings=nothing,val_encodings=nothing,tst_encodings=nothing))
                    end

                    # here define what additional info should be saved together with parameters, scores, labels and predict times
                    save_entries = merge(training_info, (modelname = modelname, seed = seed, 
                        dataset = dataset, anomaly_class = i,
                        contamination=contamination))

                    # now loop over all anomaly score funs
                    for result in results
                        GenerativeAD.experiment(result..., data, savepath; save_entries...)
                    end
                    global try_counter = max_tries + 1
                else
                    @info "Model already present, trying new hyperparameters..."
                    global try_counter += 1
                end
            end
        end
    end
    (try_counter == max_tries) ? (@info "Reached $(max_tries) tries, giving up.") : nothing
end
