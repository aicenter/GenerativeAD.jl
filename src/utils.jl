"""
    function resize_images(images, isize::Int, channels=1)

resizes whole batch of images.
"""
function resize_images(images, isize::Int, channels=1) """ zjisiti jestli to takhle vlastně jde"""
	N = size(images, 4)
	channels_list = []
	for ch=1:channels
		push!(channels_list, reshape(imresize(images[:,:,ch,:], isize, isize, N), (isize, isize, 1, N)))
	end
	return cat(channels_list..., dims=3)
end

function scale_to_interval(X; range=[0,1])
    X_std = (X .- minimum(X))./(maximum(X)-minimum(X))
    return X_std .* (range[2]-range[1]) .+ range[1]
end

"""
    function preprocess_images(data, parameters)

Preprocess image data for ganomaly.
"""
function preprocess_images(data, parameters; range=[-1,1], denominator=16)
    (X_train, y_train), (X_val, y_val), (X_test, y_test) = data
    isize = maximum([size(X_train,1),size(X_train,2)]) # there is already fixed isize in parameters
    in_ch = parameters.in_ch

    residue = isize % denominator
    if residue != 0
        isize = isize + denominator - residue
        X_train = scale_to_interval(resize_images(X_train, isize, in_ch), range=range)
        X_val = scale_to_interval(resize_images(X_val, isize, in_ch), range=range)
        X_test = scale_to_interval(resize_images(X_test, isize, in_ch), range=range)
    end

    return (X_train, y_train), (X_val, y_val), (X_test, y_test)
end

struct GANomalyHistory
    history
end

function GANomalyHistory()
    history = Dict(
        "generator_loss" => Array{Float32}([]),
        "adversarial_loss" => Array{Float32}([]),
        "contextual_loss" => Array{Float32}([]),
        "latent_loss" => Array{Float32}([]),
        "discriminator_loss" => Array{Float32}([]),
        "val_generator_loss" => Array{Float32}([]),
        "val_discriminator_loss" => Array{Float32}([])
        )
    return GANomalyHistory(history)
end

"""
	function update_history(history::GANomalyHistory, gl, dl, vgl=nothing, vdl=nothing)

do logging losses into history
"""
function update_history(history::GANomalyHistory, gl, dl, vgl=nothing, vdl=nothing)
    if gl != nothing & dl != nothing
    	push!(history["generator_loss"], gl[1])
    	push!(history["adversarial_loss"], gl[2])
    	push!(history["contextual_loss"], gl[3])
    	push!(history["encoder_loss"], gl[4])
    	push!(history["discriminator_loss"], dl)
    end
    if vgl != nothing & vdl != nothing
        push!(history["val_generator_loss"], vgl)
        push!(history["val_discriminator_loss"], vdl)
    end
	return history
end

function prepare_dataloders(data, params)
    train_loader = Flux.Data.DataLoader(data[1][1], batchsize=params.batch_size, shuffle=true)
    # for cheching convergence I need to drop anomal samples from validation data
    val_data_ind = findall(x->x==0, data[2][2])
    if length(size(data[2][1]))==4
        val_data = data[2][1][:,:,:,val_data_ind]
    else
        val_data = data[2][1][:,val_data_ind]
    end
    val_loader = Flux.Data.DataLoader(val_data, batchsize=params.batch_size)
    return train_loader, val_loader
end
