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
