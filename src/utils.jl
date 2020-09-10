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
function update_history(history::Dict{String,Array{Float32,1}}, gl::NTuple{4,Float32}, dl::Float32)
	push!(history["generator_loss"], gl[1])
	push!(history["adversarial_loss"], gl[2])
	push!(history["contextual_loss"], gl[3])
	push!(history["encoder_loss"], gl[4])
	push!(history["discriminator_loss"], dl)
	return history
end
