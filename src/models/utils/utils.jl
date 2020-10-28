using DataFrames
using CSV

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

function GANomalyHistory()
	history = Dict(
		"generator_loss" => Array{Float32}([]),
		"adversarial_loss" => Array{Float32}([]),
		"contextual_loss" => Array{Float32}([]),
		"encoder/latent_loss" => Array{Float32}([]),
		"discriminator_loss" => Array{Float32}([]),
		"val_generator_loss" => Array{Float32}([]),
		"val_discriminator_loss" => Array{Float32}([])
		)
	return history
end

"""
	function update_history(history, gl, dl, vgl=nothing, vdl=nothing)

do logging losses into history
"""
function update_history(history, gl, dl)
	push!(history["generator_loss"], gl[1])
	push!(history["adversarial_loss"], gl[2])
	push!(history["contextual_loss"], gl[3])
	push!(history["encoder/latent_loss"], gl[4])
	push!(history["discriminator_loss"], dl)
	return history
end

function update_val_history(history, vgl, vdl)
	push!(history["val_generator_loss"], vgl)
	push!(history["val_discriminator_loss"], vdl)
	return history
end

"""
	function prepare_dataloaders(data, params)

Extracts normal data from validation dataset and returns training MLDataPattern.RandomBatches and 
validation Flux.Data.DataLoader.
"""
function prepare_dataloaders(data, params)
	train_loader = MLDataPattern.RandomBatches(data[1][1], size=params.batch_size, count=params.iters)
	#train_loader = Flux.Data.DataLoader(data[1][1], batchsize=params.batch_size, shuffle=true)
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

"""
	clip_weights!(ps::Flux.Zygote.Params,c::Real)
	clip_weights!(ps::Flux.Zygote.Params,low::Real,high::Real)

Clips weights so they lie in the interval (-c,c)/(low,high).
"""
function clip_weights!(ps::Flux.Zygote.Params,low::Real,high::Real)
	@assert low <= high
	for p in ps
		T = eltype(p)
		p .= max.(p, T(low))
		p .= min.(p, T(high))
	end
end
clip_weights!(ps::Flux.Zygote.Params,c::Real) = clip_weights!(ps,-abs(c),abs(c))

"""
	helper functions for two stage models
"""
function return_best_10(df, dataset="MNIST")
    df = df[df.dataset .== dataset, :]
    top_10 = first(sort(by(df, :params, :loss_val => mean), :loss_val_mean), 10)
    df_top_10 = []
    for i=1:10
        tmp = df[df.params .== top_10.params[i],:]
        ind = i.*ones(size(tmp,1))
        push!(df_top_10, hcat(tmp, DataFrame(ind=ind)))
    end
    return vcat(df_top_10...)
end


function load_encoding(;dataset::String="MNIST", anomaly_class::Int=1, seed::Int=1, model_index::Int=1)
    # load csv
    df = CSV.read(datadir("vae_tab.csv")) #??? hardcoded
    df = return_best_10(df, dataset)
    # checking if model configuration was trained on all classes and all seeds
    n_comb = 500
    check_comb = sum([sum(df.ind .== i) for i=1:10])
    if check_comb < n_comb
        @info "One of chosen models does not include all anomaly classes and all seeds!! $(check_comb) out of $(n_comb)"
    end
    # get correct model path
    encoding_path = df[(df.ac .== anomaly_class) .& (df.seed .==1) .& (df.ind .==model_index),:][:path]
    
	model = BSON.load(encoding_path)
	
    return (model[:tr_encodings], model[:val_encodings], model[:test_encodings]), split(encoding_path, "/")[end]
end