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

"""
    function update_history(history::Dict{String,Array{Float32,1}}, gl::NTuple{4,Float32}, dl::Float32)

do logging losses into history
"""
function update_history(history::Dict{String,Array{Float32,1}}, gl, dl)
	push!(history["generator_loss"], gl[1]|>cpu)
	push!(history["adversarial_loss"], gl[2]|>cpu)
	push!(history["contextual_loss"], gl[3]|>cpu)
	push!(history["encoder_loss"], gl[4]|>cpu)
	push!(history["discriminator_loss"], dl|>cpu)
	return history
end


"""
    function preprocess_images(data, parameters)

Preprocess image data for ganomaly.
"""
function preprocess_images(data, parameters)
    (X_train, y_train), (X_val, y_val), (X_test, y_test) = data
    isize = maximum([size(X_train,1),size(X_train,2)]) # there is already fixed isize in parameters
    in_ch = parameters.in_ch

    residue = isize % 16
    if residue != 0
        isize = isize + 16 - residue
        X_train = resize_images(X_train, isize, in_ch)
        X_val = resize_images(X_val, isize, in_ch)
        X_test = resize_images(X_test, isize, in_ch)
    end
    return (X_train, y_train), (X_val, y_val), (X_test, y_test)
end
