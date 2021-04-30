using DrWatson
@quickactivate
using ArgParse
using GenerativeAD
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
    "contamination"
        arg_type = Float64
        help = "contamination rate of training data"
        default = 0.0
end
parsed_args = parse_args(ARGS, s)
@unpack dataset, max_seed, contamination = parsed_args

modelname = "dagmm"

function sample_params()
    parameter_rng = (
        zdim 		= 2 .^(0:2),
        hdim 		= 2 .^(1:8),
        nlayers 	= 1:3,
        ncomp       = 2:8,
        lambda_rat 	= 1:9,
        lr 			= [1f-3, 1f-4, 1f-5],
        batchsize 	= 2 .^ (5:8),
        activation	= ["tanh"],
        dropout		= 0.0:0.1:0.5,
        wreg 		= [0.0, 1f-5, 1f-6],
        init_seed 	= 1:Int(1e8),
    )

    (;zip(keys(parameter_rng), map(x->sample(x, 1)[1], parameter_rng))...)
end

function fit(data, parameters)
    model = GenerativeAD.Models.DAGMM(;idim=size(data[1][1], 1), parameters...)

    try
        global info, fit_t, _, _, _ = @timed fit!(model, data; max_train_time=82800/max_seed, 
                        patience=200, check_interval=1, parameters...)
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
    testmode!(model, true)
    _, _, z, gamma = model(data[1][1])
    phi, mu, cov = GenerativeAD.Models.compute_params(z, gamma)
    testmode!(model, false)

    # this will have to store the parameters obtained from clean data
    training_info, [(x -> predict(info.model, x, phi, mu, cov), parameters)]
end


try_counter = 0
max_tries = 10*max_seed
cont_string = (contamination == 0.0) ? "" : "_contamination-$contamination"
while try_counter < max_tries
    parameters = sample_params()

    for seed in 1:max_seed
        savepath = datadir("experiments/tabular$cont_string/$(modelname)/$(dataset)/seed=$(seed)")
        mkpath(savepath)

        # get data
        data = GenerativeAD.load_data(dataset, seed=seed, contamination=contamination)
        edited_parameters = GenerativeAD.edit_params(data, parameters)

        if GenerativeAD.check_params(savepath, edited_parameters)
            @info "Started training $(modelname)$(edited_parameters) on $(dataset):$(seed)"
            @info "Train/valdiation/test splits: $(size(data[1][1], 2)) | $(size(data[2][1], 2)) | $(size(data[2][1], 2))"
            @info "Number of features: $(size(data[1][1], 1))"
            
            training_info, results = fit(data, edited_parameters)

            if training_info.model != nothing
                tagsave(joinpath(savepath, savename("model", edited_parameters, "bson", digits=5)), 
                        Dict("model"=>training_info.model,
                            "fit_t"=>training_info.fit_t,
                            "history"=>training_info.history,
                            "parameters"=>edited_parameters
                            ), safe = true)
                training_info = merge(training_info, (model = nothing,))
            end
            save_entries = merge(training_info, (modelname = modelname, seed = seed, dataset = dataset, contamination = contamination))

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
(try_counter == max_tries) ? (@info "Reached $(max_tries) tries, giving up.") : nothing
