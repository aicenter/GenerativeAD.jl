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
		nothing : sublib.download(i_accept_the_terms_of_use=true)

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
