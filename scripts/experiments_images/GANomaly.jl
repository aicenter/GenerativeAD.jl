using ArgParse
using GenerativeAD
using DrWatson
@quickactivate
using BSON
using StatsBase: fit!, predict

using Flux
using MLDataPattern

s = ArgParseSettings()
@add_arg_table! s begin
    "dataset"
        required = true
        arg_type = String
        help = "dataset"
    "seed"
        required = true
        arg_type = Int
        help = "seed"
end
parsed_args = parse_args(ARGS, s)
@unpack dataset, seed = parsed_args


modelname = "Conv-GANomaly"


"""
    function fit(data, parameters)

parameters => type named tuple with keys
    latent_dim    - dimension of latent space on the encoder's end
    num_filters   - number of kernels/masks in convolutional layers
    extra_layers  - number of additional conv layers
    lr            - learning rate for optimiser
    iters         - number of optimisation steps (iterations) during training
    batch_size    - batch/minibatch size

Note:
    data = load_data("MNIST")
    (x_train, y_train), (x_val, y_val), (x_test, y_test) = data
"""
function fit(data, parameters)
    # data preprocessing
    (X_train,_), (X_val, y_val), (X_test, y_test) = data

    # computing additional parameters and resizing input data
    in_ch = out_ch = size(X_train,3)
    isize = maximum([size(X_train,1),size(X_train,2)])
    residue = isize % 16
    if residue != 0
        isize = isize + 16 - residue
        X_train = GenerativeAD.resize_images(X_train, isize, in_ch)
        X_val = GenerativeAD.resize_images(X_val, isize, in_ch)
        X_test = GenerativeAD.resize_images(X_test, isize, in_ch)
    end
    # prepare batches & loaders
    train_loader = MLDataPattern.RandomBatches(X_train, parameters.batch_size, parameters.iters)
    #valid_loader = Flux.Data.DataLoader((X_val,y_val), batchsize=parameters.batch_size, shuffle=false)
    #test_loader = Flux.Data.DataLoader((X_test, y_test), batchsize=parameters.batch_size, shuffle=false)

    # define models (Generator, Discriminator)
    generator = GenerativeAD.Models.ConvGenerator(isize, parameters.latent_dim, in_ch, parameters.num_filters, parameters.extra_layers)
    discriminator = GenerativeAD.Models.ConvDiscriminator(isize, in_ch, out_ch, parameters.num_filters, parameters.extra_layers)

    # define optimiser
    opt = Flux.Optimise.ADAM(parameters.lr)

    try
		global info, fit_t, _, _, _ = @timed fit!(generator, discriminator, opt, train_loader)
	catch e
		return Dict(:fit_t => NaN), (nothing, nothing)
	end

    training_info = Dict(
		:fit_t => fit_t,
        :history => info[1]
		)

    return training_info, predict(X_val)

end
