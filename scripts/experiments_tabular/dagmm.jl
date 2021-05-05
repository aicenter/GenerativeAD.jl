using DrWatson
@quickactivate
using ArgParse
using GenerativeAD
using PyCall
using StatsBase: fit!, predict, sample
using BSON
using Flux

s = ArgParseSettings()
@add_arg_table! s begin
    "max_seed"
        default = 1
        arg_type = Int
        help = "maximum number of seeds to run through"
    "dataset"
        default = "iris"
        arg_type = String
        help = "dataset"
    "sampling"
        default = "random"
        arg_type = String
        help = "sampling of hyperparameters"
    "contamination"
        arg_type = Float64
        help = "contamination rate of training data"
        default = 0.0
end
parsed_args = parse_args(ARGS, s)
@unpack dataset, max_seed, sampling, contamination = parsed_args

modelname = "dagmm"

function sample_params()
    parameter_rng = (
        zdim 		= 2 .^(0:1),
        hdim 		= 2 .^(1:8),
        nlayers 	= 2:3,
        ncomp       = 2:8,
        lambda_e 	= [1.0, 0.5, 0.1],
        lambda_d    = [1.0, 0.5, 0.1, 0.05, 0.005],
        lr 			= [1f-3, 1f-4, 1f-5],
        batchsize 	= 2 .^ (5:8),
        activation	= ["tanh"],
        dropout		= 0.0:0.1:0.5,
        wreg 		= [0.0],
        init_seed 	= 1:Int(1e8),
    )

    (;zip(keys(parameter_rng), map(x->sample(x, 1)[1], parameter_rng))...)
end

function create_space()
    pyReal = pyimport("skopt.space")["Real"]
    pyInt = pyimport("skopt.space")["Integer"]
    pyCat = pyimport("skopt.space")["Categorical"]
    
    (;  
        zdim 		= pyInt(0, 1, 								name="log2_zdim"),
        hdim 		= pyInt(1, 8, 								name="log2_hdim"),
        nlayers 	= pyInt(2, 3, 								name="nlayers"),
        ncomp       = pyInt(2, 8, 								name="ncomp"),
        lambda_e 	= pyReal(1e-2, 1.0, prior="log-uniform", 	name="lambda_e"),
        lambda_d    = pyReal(1e-3, 1.0, prior="log-uniform", 	name="lambda_d"),
        lr 			= pyReal(1f-5, 1f-3, prior="log-uniform", 	name="lr"),
        batchsize 	= pyInt(5, 7, 								name="log2_batchsize"),
        activation	= pyCat(categories=["tanh"], 		        name="activation"),
        dropout		= pyReal(0.0, 0.5,                          name="dropout"),
        wreg 		= pyCat(categories=[0.0], 			        name="wreg"),
    )
end

function fit(data, parameters)
    model = GenerativeAD.Models.DAGMM(;idim=size(data[1][1], 1), parameters...)

    try
        global info, fit_t, _, _, _ = @timed fit!(model, data; max_train_time=82800/max_seed, 
                        patience=20, check_interval=10, parameters...)
    catch e
        @info "Failed training due to \n$e"
        return (fit_t = NaN, history=nothing, npars=nothing, model=nothing), []
    end

    training_info = (
        fit_t = fit_t,
        history = info.history,
        niter = info.niter,
        npars = info.npars,
        model = info.model
        )

    # compute mixture parameters from the whole training data
    # might be better to do it with batches at some point
    testmode!(model, true)
    _, _, z, gamma = model(data[1][1])
    phi, mu, cov = GenerativeAD.Models.compute_params(z, gamma)
    testmode!(model, false)

    # this will have to store the parameters obtained from clean data
    training_info, [(x -> predict(info.model, x, phi, mu, cov), parameters)]
end

function GenerativeAD.edit_params(data, parameters)
    idim = size(data[1][1],1)
    # set hdim ~ idim/2 if hdim >= idim
    if parameters.hdim >= idim
        hdims = 2 .^(1:8)
        hdim_new = hdims[hdims .<= (idim+1)//2][end]
        @info "Lowering width of autoencoder $(parameters.hdim) -> $(hdim_new)"
        parameters = merge(parameters, (hdim=hdim_new,))
    end
    parameters
end

try_counter = 0
max_tries = 10*max_seed
cont_string = (contamination == 0.0) ? "" : "_contamination-$contamination"
sampling_string = sampling == "bayes" ? "_bayes" : ""
prefix = "experiments$(sampling_string)/tabular$(cont_string)"
dataset_folder = datadir("$(prefix)/$(modelname)/$(dataset)")
while try_counter < max_tries
    if sampling == "bayes"
        parameters = GenerativeAD.bayes_params(
                                create_space(), 
                                dataset_folder,
                                sample_params; add_model_seed=true)
    else
        parameters = sample_params()
    end

    for seed in 1:max_seed
        savepath = joinpath(dataset_folder, "seed=$(seed)")
        mkpath(savepath)

        # get data
        data = GenerativeAD.load_data(dataset, seed=seed, contamination=contamination)
        edited_parameters = sampling == "bayes" ? parameters : GenerativeAD.edit_params(data, parameters)

        if GenerativeAD.check_params(savepath, edited_parameters)
            @info "Started training $(modelname)$(edited_parameters) on $(dataset):$(seed)"
            @info "Train/valdiation/test splits: $(size(data[1][1], 2)) | $(size(data[2][1], 2)) | $(size(data[2][1], 2))"
            @info "Number of features: $(size(data[1][1], 1))"
            
            training_info, results = fit(data, edited_parameters)

            if training_info.model !== nothing
                tagsave(joinpath(savepath, savename("model", edited_parameters, "bson", digits=5)), 
                        Dict("model"=>training_info.model,
                            "fit_t"=>training_info.fit_t,
                            "history"=>training_info.history,
                            "parameters"=>edited_parameters
                            ), safe = true)
                training_info = merge(training_info, (model = nothing,))
            end
            save_entries = merge(training_info, (modelname = modelname, seed = seed, dataset = dataset, contamination = contamination))

            all_scores = [GenerativeAD.experiment(result..., data, savepath; save_entries...) for result in results]
            if sampling == "bayes" && length(all_scores) > 0
                @info("Updating cache with $(length(all_scores)) results.")
                GenerativeAD.update_bayes_cache(dataset_folder, 
                        all_scores; ignore=Set([:init_seed]))
            end
            global try_counter = max_tries + 1
        else
            @info "Model already present, trying new hyperparameters..."
            global try_counter += 1
        end
    end
end
(try_counter == max_tries) ? (@info "Reached $(max_tries) tries, giving up.") : nothing
