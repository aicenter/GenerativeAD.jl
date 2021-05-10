using DrWatson
@quickactivate
using ArgParse
using GenerativeAD
using PyCall
using OrderedCollections
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
        help = "sampling of hyperparameters [bayes|random]"
    "contamination"
        arg_type = Float64
        help = "contamination rate of training data"
        default = 0.0
end
parsed_args = parse_args(ARGS, s)
@unpack dataset, max_seed, sampling, contamination = parsed_args

modelname = "repen"

function sample_params()
    parameter_rng = (
        zdim            = 2 .^(1:6),
        hdim            = 2 .^(1:8),
        conf_margin     = 2 .^(9:11),
        nlayers         = 1:2,
        ensemble_size   = [25, 50],
        subsample_size  = 2 .^(1:8),
        batchsize       = 2 .^ (5:8),
        activation      = ["relu", "tanh"],
        init_seed       = 1:Int(1e8)
    )

    (;zip(keys(parameter_rng), map(x->sample(x, 1)[1], parameter_rng))...)
end

function create_space()
    pyReal = pyimport("skopt.space")["Real"]
    pyInt = pyimport("skopt.space")["Integer"]
    pyCat = pyimport("skopt.space")["Categorical"]
    
    (;
        zdim            = pyInt(0, 6, 							name="log2_zdim"),
        hdim            = pyInt(1, 8, 							name="log2_hdim"),
        conf_margin     = pyInt(250, 2050,                      name="conf_margin"),
        nlayers         = pyInt(1, 2, 							name="nlayers"),
        ensemble_size   = pyCat(categories=[25, 50], 		    name="ensemble_size"),
        subsample_size  = pyInt(1, 8, 							name="log2_subsample_size"),
        batchsize       = pyInt(3, 8, 							name="log2_batchsize"),
        activation      = pyCat(categories=["relu", "tanh"], 	name="activation"),
    )
end

function fit(data, parameters)
    model = GenerativeAD.Models.REPEN(;idim=size(data[1][1], 1), parameters...)

    try
        global info, fit_t, _, _, _ = @timed fit!(model, data; 
                                max_train_time=82800/max_seed, parameters...)
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

    training_info, [(x -> predict(info.model, x; fit_data=data[1][1], parameters...), parameters)]
end

function GenerativeAD.edit_params(data, parameters)
    idim, n = size(data[1][1])
    
    if parameters.nlayers > 1
        # set hdim ~ idim/2 if hdim >= idim
        if parameters.hdim >= idim
            hdims = 2 .^(1:8)
            hdim_new = hdims[hdims .<= (idim+1)//2][end]
            @info "width of hidden layer hdim $(parameters.hdim) too large for input dimension $(idim) lowering -> $(hdim_new)"
            parameters = merge(parameters, (hdim=hdim_new,))
        end

        # ensure that zdim < hdim
        while parameters.zdim >= parameters.hdim
            new_zdim = sample(2 .^(0:6))
            @info "zdim $(parameters.zdim) too large in combination with hdim $(parameters.hdim) trying $(new_zdim)"
            parameters = merge(parameters, (;zdim = new_zdim))
        end
    else
        # ensure that zdim < idim
        while parameters.zdim >= idim
            new_zdim = sample(2 .^(0:6))
            @info "zdim $(parameters.zdim) too large for input dimension $(idim) trying $(new_zdim)"
            parameters = merge(parameters, (;zdim = new_zdim))
        end
    end

    # modify batchsize to < n/3
    if parameters.batchsize >= n//3
        batchsizes = 2 .^(1:8)
        batchsize_new = batchsizes[batchsizes .< n//3][end]
        @info "Decreasing batchsize due to small number of samples $(parameters.batchsize) -> $(batchsize_new)"
        parameters = merge(parameters, (batchsize=batchsize_new,)) 
    end

    # modify subsample_size to < n
    while parameters.subsample_size >= n
        subsample_new = sample(2 .^(1:8))
        @info "subsample_size $(parameters.subsample_size) too large for number of samples $(n) trying $(subsample_new)"
        parameters = merge(parameters, (subsample_size=subsample_new,)) 
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
        # edit_params is not deterministic and therefore `edited_parameters` may be different for each seed
        # as a workaround once it is called it overwrites `parameters`, which should not be changed by the next call
        if sampling == "bayes"
            edited_parameters = parameters
        else
            edited_parameters = parameters = GenerativeAD.edit_params(data, parameters)
        end

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
