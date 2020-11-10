"""
    train_val_test_inds(indices, ratios=(0.6,0.2,0.2); seed=nothing)

Split indices.
"""
function train_val_test_inds(indices, ratios=(0.6,0.2,0.2); seed=nothing)
    (sum(ratios) == 1 && length(ratios) == 3) ? nothing :
    	error("ratios must be a vector of length 3 that sums up to 1")

    # set seed
    (seed == nothing) ? nothing : Random.seed!(seed)

    # set number of samples in individual subsets
    n = length(indices)
    ns = cumsum([x for x in floor.(Int, n .* ratios)])

    # scramble indices
    _indices = sample(indices, n, replace=false)

    # restart seed
    (seed == nothing) ? nothing : Random.seed!()

    # return the sets of indices
    _indices[1:ns[1]], _indices[ns[1]+1:ns[2]], _indices[ns[2]+1:ns[3]]
end

"""
	train_val_test_split(data_normal, data_anomalous, ratios=(0.6,0.2,0.2); seed=nothing,
	    	contamination::Real=0.0)

Split data.
"""
function train_val_test_split(data_normal, data_anomalous, ratios=(0.6,0.2,0.2); seed=nothing,
	    	contamination::Real=0.0)
	# split the normal data, add some anomalies to the train set and divide
	# the rest between validation and test
	(0 <= contamination <= 1) ? nothing : error("contamination must be in the interval [0,1]")
	nd = ndims(data_normal) # differentiate between 2D tabular and 4D image data

	# split normal indices
	indices = 1:size(data_normal, nd)
	split_inds = train_val_test_inds(indices, ratios; seed=seed)

	# select anomalous indices
	indices_anomalous = 1:size(data_anomalous, nd)
	vtr = (1 - contamination)/2 # validation/test ratio
	split_inds_anomalous = train_val_test_inds(indices_anomalous, (contamination, vtr, vtr); seed=seed)

	# this can be done universally - how?
	if nd == 2
		tr_n, val_n, tst_n = map(is -> data_normal[:,is], split_inds)
		tr_a, val_a, tst_a = map(is -> data_anomalous[:,is], split_inds_anomalous)
	elseif nd == 4
		tr_n, val_n, tst_n = map(is -> data_normal[:,:,:,is], split_inds)
		tr_a, val_a, tst_a = map(is -> data_anomalous[:,:,:,is], split_inds_anomalous)
	end

	# cat it together
	tr_x = cat(tr_n, tr_a, dims = nd)
	val_x = cat(val_n, val_a, dims = nd)
	tst_x = cat(tst_n, tst_a, dims = nd)

	# now create labels
	tr_y = vcat(zeros(Float32, size(tr_n, nd)), ones(Float32, size(tr_a,nd)))
	val_y = vcat(zeros(Float32, size(val_n, nd)), ones(Float32, size(val_a,nd)))
	tst_y = vcat(zeros(Float32, size(tst_n, nd)), ones(Float32, size(tst_a,nd)))

	(tr_x, tr_y), (val_x, val_y), (tst_x, tst_y)
end


"""
	load_data(dataset::String, ratios=(0.6,0.2,0.2); seed=nothing, contamination::Real=0.0)

Returns 3 tuples of (data, labels) representing train/validation/test part. Arguments are the splitting
ratios for normal data, seed and training data contamination.

For a list of available datasets, check `GenerativeAD.Datasets.uci_datasets`, `GenerativeAD.Datasets.other_datasets`
and `GenerativeAD.Datasets.mldatasets`.
"""
function load_data(dataset::String, ratios=(0.6,0.2,0.2); seed=nothing, contamination::Real=0.0, kwargs...)
	# extract data and labels
	if dataset in uci_datasets # UCI Loda data, standardized
		data_normal, data_anomalous = load_uci_data(dataset; kwargs...)
	elseif dataset in mldatasets # MNIST,FMNIST, SVHN2, CIFAR10
		data_normal, data_anomalous = load_mldatasets_data(dataset; kwargs...)
	elseif dataset in other_datasets
		data_normal, data_anomalous = load_other_data(dataset; standardize=true, kwargs...)
	elseif dataset in mill_datasets
		data_normal, data_anomalous = load_mill_data(dataset; kwargs...)
	else
		error("Dataset $(dataset) not known, either not implemented or misspeled.")
		# TODO add the rest
	end

	# now do the train/validation/test split
	train_val_test_split(data_normal, data_anomalous, ratios; seed=seed, contamination=contamination)
end


function load_datam(dataset::String, ratios=(0.6,0.2,0.2); seed=nothing, contamination::Real=0.0, kwargs...)
	# extract data and labels
	data_normal, data_anomalous = load_mill_data(dataset; kwargs...)
	# now do the train/validation/test split
	train_val_test_split(data_normal, data_anomalous, ratios; seed=seed, contamination=contamination)
end

"""
	vectorize(data)

Vectorizes the 4D image data returned from `load_data`.
"""
vectorize(data) = map(d->(reshape(d[1], :, size(d[1],4)), d[2]),data)
