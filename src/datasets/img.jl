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
	load_mvtec_ad_data(;category::String="bottle")

Loads the corrupted MNIST dataset for given category. Returns normal (non-corrupted) and anomalous data. For 
a list of available corruption categories, run `GenerativeAD.Datasets.mnist_c_categories()`.
"""
function load_mvtec_ad_data(;category::String="bottle", kwargs...)
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
	load_mvtec_ad_data_downscaled(;category::String="wood")

Returns the normal and anomalous MVTec-AD data for a given category.
"""
function load_mvtec_ad_data_downscaled(;category::String="bottle", kwargs...)
	inpath = datadir("mvtec_ad/downscaled_data")
	!ispath(inpath) ? error("downscaled data not available at $inpath") : nothing
	data = load(joinpath(inpath, "$(category).bson"))
	return data[:normal], data[:anomalous]
end

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
