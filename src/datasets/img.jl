mldatasets = ["MNIST", "FashionMNIST", "CIFAR10", "SVHN2"]

"""
	load_mldatasets_data(dataset::String[; anomaly_class_ind=1])

Loads MNIST, FMNIST, SVHN2 and CIFAR10 datasets.
"""
function load_mldatasets_data(dataset::String; anomaly_class_ind::Int=1)
	(dataset in mldatasets) ? nothing : error("$dataset not available in MLDatasets.jl")
	# since the functions for MLDatasets.MNIST, MLDatasets.CIFAR10 are the same
	sublib = getfield(MLDatasets, Symbol(dataset))

	# do we need to download it?
	isdir(joinpath(first(DataDeps.standard_loadpath), dataset)) ?
		nothing : sublib.download()

	# now get the data
	tr_x, tst_x = sublib.traintensor(Float32), sublib.testtensor(Float32)
	if ndims(tr_x) == 3 # some datasets are 3D tensors :(
		tr_x = reshape(tr_x, size(tr_x,1), size(tr_x,2), 1, :)
		tst_x = reshape(tst_x, size(tst_x,1), size(tst_x,2), 1, :)
	end
	data = cat(tr_x, tst_x, dims=4)
	labels = cat(sublib.trainlabels(), sublib.testlabels(), dims=1)

	# return the normal and anomalous data
	label_list = unique(labels)
	aclass_inds = labels .== label_list[anomaly_class_ind] # indices of anomalous data
	data[:,:,:,.!aclass_inds], data[:,:,:,aclass_inds]
end

"""
	mnist_c_categories()

List of the available MNIST-C categories.
"""
mnist_c_categories() = readdir(joinpath(datadep"MNIST-C", "mnist_c"))

"""
	load_mnist_c_data(;category::String="brightness")

Loads the corrupted MNIST dataset for given category. Returns normal (non-corrupted) and anomalous data. For 
a list of available corruption categories, run `GenerativeAD.Datasets.mnist_c_categories()`.
"""
function load_mnist_c_data(;category::String="brightness", kwargs...)
	# TODO the flip here should be (3,2,4,1) so its is the same as the colored datasets
	# or not? check with the rest of the bw datasets
	dp = joinpath(datadep"MNIST-C", "mnist_c")
	available_categories = mnist_c_categories()
	!(category in available_categories) ? error("Requested category $category not found, $(available_categories) available.") : nothing
	tr_x_a, tst_x_a = map(x->Float32.(permutedims(npzread(joinpath(dp, category, x))/255, (2,3,4,1))), 
		["train_images.npy", "test_images.npy"])
	tr_x_n, tst_x_n = map(x->Float32.(permutedims(npzread(joinpath(dp, "identity", x))/255, (2,3,4,1))), 
		["train_images.npy", "test_images.npy"])
	return cat(tr_x_n, tst_x_n, dims=4), cat(tr_x_a, tst_x_a, dims=4)
end

"""
	mvtec_ad_categories()

List of the available MVTec-AD categories.
"""
mvtec_ad_categories() = readdir(datadep"MVTec-AD")

"""
	load_mvtec_ad_data(;category::String="bottle", img_size=nothing)

Loads the corrupted MNIST dataset for given category. Returns normal (non-corrupted) and anomalous data. For 
a list of available corruption categories, run `GenerativeAD.Datasets.mnist_c_categories()`.
"""
function load_mvtec_ad_data(;category::String="bottle", img_size=nothing, kwargs...)
	# return the downscaled version if needed
	isnothing(img_size) ? nothing : 
		return load_mvtec_ad_data_downscaled(category=category, img_size=img_size)
	# else get the original
	dp = datadep"MVTec-AD"
	available_categories = mvtec_ad_categories()
	!(category in available_categories) ? error("Requested category $category not found, $(available_categories) available.") : nothing
	cdp = joinpath(dp, category)
	# training and testing normal images
	tr_imgs_n = load_images_rgb(joinpath(cdp, "train/good"))
	tst_imgs_n = load_images_rgb(joinpath(cdp, "test/good"))
	# anomalous images
	tst_imgs_a = map(x->load_images_rgb(joinpath(cdp, "test", x)), filter(x->x!="good", readdir(joinpath(cdp, "test"))))

	return cat(tr_imgs_n, tst_imgs_n, dims=4), cat(tst_imgs_a..., dims=4)
end

"""
	load_mvtec_ad_data_downscaled(;img_size = 256, category::String="wood")

Returns the normal and anomalous MVTec-AD data for a given category.
"""
function load_mvtec_ad_data_downscaled(;img_size = 256, category::String="bottle", kwargs...)
	inpath = datadir("mvtec_ad/downscaled_data")
	!ispath(inpath) ? error("downscaled data not available at $inpath") : nothing
	data = load(joinpath(inpath, "$(category)_$(img_size).bson"))
	return data[:normal], data[:anomalous]
end

function load_multifactor_data(x_tr, y_tr, x_tst, y_tst; 
	normal_class_ind::Union{Int,Tuple,Vector}=1,
	anomaly_class_ind::Union{Int,Tuple,Vector}=-1,
	denormalize = false,
	kwargs...)
	# denormalize
	if denormalize
		x_tr = x_tr .* 0.5f0 .+ 0.5f0
		x_tst = x_tst .* 0.5f0 .+ 0.5f0
	end

	# now get the normal and anomalous data
	function compare_inds(y::Vector, ind::Int)
		if ind == -1
			return BitArray{1}(ones(length(y)))
		else
			return y .== ind
		end
	end

	# normal class
	# specified by a scalar
	if typeof(normal_class_ind) <: Number
		nc_inds = y_tr .== normal_class_ind
		xn = x_tr
		yn = y_tr[nc_inds]
	# specified by a vector - multiple digits 
	elseif typeof(normal_class_ind) <: Vector
		nc_inds = map(x->x in normal_class_ind, y_tr)
		xn = x_tr
		yn = y_tr[nc_inds]
	# specified by a tuple - we draw from the mixed dataset, -1 in the tuple means the factor
	# is not fixed
	else
		length(normal_class_ind) == 3 ? nothing : error("Normal class tuple has to be of length 3.")
		nc_inds = map(i->compare_inds(y_tst[i,:], normal_class_ind[i]), 1:3)
		nc_inds = nc_inds[1] .& (nc_inds[2] .& nc_inds[3])
		xn = x_tst
		yn = y_tst[:,nc_inds]
	end

	# anomaly data
	# default
	if anomaly_class_ind == -1
		ac_inds = .!nc_inds
		if ndims(yn) == 1
			xa = x_tr
			ya = y_tr[ac_inds]
		else
			xa = x_tst
			ya = y_tst[:, ac_inds]
		end
	# non-default integer - only one anomaly class that has the same factors
	elseif typeof(anomaly_class_ind) <: Number
		ac_inds = y_tr .== anomaly_class_ind
		xa = x_tr
		ya = y_tr[ac_inds]
	# specified by a vector - multiple digits
	elseif typeof(anomaly_class_ind) <: Vector
		ac_inds = map(x->x in anomaly_class_ind, y_tr)
		xa = x_tr
		ya = y_tr[ac_inds]
	# specified by a tuple - we draw from the mixed dataset, -1 in the tuple means the factor
	# is not fixed
	else
		length(anomaly_class_ind) == 3 ? nothing : error("Anomaly class tuple has to be of length 3.")
		ac_inds = map(i->compare_inds(y_tst[i,:], anomaly_class_ind[i]), 1:3)
		ac_inds = ac_inds[1] .& (ac_inds[2] .& ac_inds[3])
		xa = x_tst
		ya = y_tst[:,ac_inds]
	end

	return (xn[:,:,:,nc_inds], yn), (xa[:,:,:,ac_inds], ya)
end

"""
	load_wildlife_mnist_raw(selection="all")

Selection is one of ["all", "train", "test"].
"""
function load_wildlife_mnist_raw(selection="all")
	inpath = datadir("wildlife_MNIST")
	if selection == "train"
		x_tr = permutedims(npzread(joinpath(inpath, "data.npy")), (4,3,2,1))
		y_tr = npzread(joinpath(inpath, "labels.npy")) .+ 1
		x_tst = y_tst = nothing
	elseif selection == "test"
		x_tr = y_tr = nothing
		x_tst = permutedims(npzread(joinpath(inpath, "data_test.npy")), (4,3,2,1))
		y_tst = Array(npzread(joinpath(inpath, "labels_test.npy"))' .+ 1)
	elseif selection == "all"
		x_tr = permutedims(npzread(joinpath(inpath, "data.npy")), (4,3,2,1))
		y_tr = npzread(joinpath(inpath, "labels.npy")) .+ 1
		x_tst = permutedims(npzread(joinpath(inpath, "data_test.npy")), (4,3,2,1))
		y_tst = Array(npzread(joinpath(inpath, "labels_test.npy"))' .+ 1)
	else
		error("unknown value $selection")
	end
	return (x_tr, y_tr), (x_tst, y_tst)
end

"""
	load_wildlife_mnist_data(;
	normal_class_ind::Union{Int,Tuple,Vector}=1,
	anomaly_class_ind::Union{Int,Tuple,Vector}=-1,
	denormalize = false,
	kwargs...)

Returns (x_normal, y_normal), (x_anomalous, y_anomalous).

If the class index is an integer, then we draw samples with all fixed factors equal to the index.
If it is a vector (with values in range [1,10]), then the samples have fixed factors in the given values.
If it is a tuple of length 3, then the factors are mixed as given. You can input -1 in the place
of the factor that is not fixed. Factor 1 = digit, factor 2 = background, factor 3 = texture.
If no anomalous indices are given, then the rest that is left after normal data is constructed is used.

The data are normalized in [-1,1] range by default, but can be denormalized to [0,1].
"""
function load_wildlife_mnist_data(; 
	normal_class_ind::Union{Int,Tuple,Vector}=1,
	anomaly_class_ind::Union{Int,Tuple,Vector}=-1,
	denormalize = false,
	kwargs...)
	# load all data
	(x_tr, y_tr), (x_tst, y_tst) = load_wildlife_mnist_raw("all")

	return load_multifactor_data(x_tr, y_tr, x_tst, y_tst; 
		normal_class_ind=normal_class_ind,
		anomaly_class_ind=anomaly_class_ind,
		denormalize=denormalize,
		kwargs...)
end

"""
	load_cocoplaces_raw(imsize=64,selection="all")

Selection is one of ["all", "uniform", "mashed"].
"""
function load_cocoplaces_raw(imsize=64,selection="all")
	inpath = datadir("cocoplaces")
	if selection == "uniform"
		x_u = permutedims(npzread(joinpath(inpath, "uniform_data_$(imsize).npy")), (4,3,2,1))
		y_u = Array(npzread(joinpath(inpath, "uniform_labels_$(imsize).npy")) .+ 1)
		x_m = y_m = nothing
	elseif selection == "test"
		x_u = y_u = nothing
		x_m = permutedims(npzread(joinpath(inpath, "mashed_data_$(imsize).npy")), (4,3,2,1))
		y_m = Array(npzread(joinpath(inpath, "mashed_labels_$(imsize).npy")) .+ 1)
	elseif selection == "all"
		x_u = permutedims(npzread(joinpath(inpath, "uniform_data_$(imsize).npy")), (4,3,2,1))
		y_u = Array(npzread(joinpath(inpath, "uniform_labels_$(imsize).npy")) .+ 1)
		x_m = permutedims(npzread(joinpath(inpath, "mashed_data_$(imsize).npy")), (4,3,2,1))
		y_m = Array(npzread(joinpath(inpath, "mashed_labels_$(imsize).npy")) .+ 1)
	else
		error("unknown value $selection")
	end
	return (x_u, y_u), (x_m, y_m)
end

"""
	load_cocoplaces_data(;
	normal_class_ind::Union{Int,Tuple,Vector}=1,
	anomaly_class_ind::Union{Int,Tuple,Vector}=-1,
	denormalize = false,
	kwargs...)

Returns (x_normal, y_normal), (x_anomalous, y_anomalous).

If the class index is an integer, then we draw samples with all fixed factors equal to the index.
If it is a vector (with values in range [1,10]), then the samples have fixed factors in the given values.
If it is a tuple of length 3, then the factors are mixed as given. You can input -1 in the place
of the factor that is not fixed. Factor 1 = digit, factor 2 = background, factor 3 = texture.
If no anomalous indices are given, then the rest that is left after normal data is constructed is used.
"""
function load_cocoplaces_data(; 
	normal_class_ind::Union{Int,Tuple,Vector}=1,
	anomaly_class_ind::Union{Int,Tuple,Vector}=-1,
	kwargs...)
	# load all data
	(x_u, y_u), (x_m, y_m) = load_cocoplaces_raw("all")
	y_u = cat(y_u, y_u[:,1:1],dims=2)
	y_m = cat(y_m, y_m[:,1:1],dims=2)

	return load_multifactor_data(x_u, y_u, x_m, y_m; 
		normal_class_ind=normal_class_ind,
		anomaly_class_ind=anomaly_class_ind,
		denormalize=false,
		kwargs...)
end

"""
	array_to_img_rgb(arr)	

Converts a 3D array (w,h,c) to an image you can display with display().
"""
array_to_img_rgb(arr) = RGB.(arr[:,:,1], arr[:,:,2], arr[:,:,3])

"""
	img_to_array_rgb(img)

Convert RGB img to a 3D tensor.
"""
img_to_array_rgb(img) = permutedims(Float32.(channelview(RGB.(img))), (2,3,1))

"""
	load_images_rgb(path)

Load all images in a path.
"""
function load_images_rgb(path)
	imgs = []
	for f in readdir(path, join=true)
		try # some files are corrupted
			img = img_to_array_rgb(load(f))
			push!(imgs, img)
		catch
			nothing
		end
	end
	cat(imgs..., dims=4)
end
