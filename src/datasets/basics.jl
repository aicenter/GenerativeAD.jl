"""
    train_val_test_inds(indices, ratios=(0.6,0.2,0.2); seed=nothing)

Split indices.
"""
function train_val_test_inds(indices, ratios=(0.6,0.2,0.2); seed=nothing)
    (sum(ratios) ≈ 1 && length(ratios) == 3) ? nothing :
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
function train_val_test_split(data_normal, data_anomalous, ratios=(0.6,0.2,0.2); 
    seed=nothing, contamination::Real=0.0)

    # split the normal data, add some anomalies to the train set and divide
    # the rest between validation and test
    (0 <= contamination <= 1) ? nothing : error("contamination must be in the interval [0,1]")
    nd = ndims(data_normal) # differentiate between 2D tabular and 4D image data

    # split normal indices
    indices = 1:size(data_normal, nd)
    split_inds = train_val_test_inds(indices, ratios; seed=seed)

    # select anomalous indices
    na = size(data_anomalous, nd)
    indices_anomalous = 1:na
    na_tr = floor(Int, length(split_inds[1])*contamination/(1-contamination))
    (na_tr > na) ? error("selected contamination rate $contamination is too high, not enough anomalies available") : nothing
    tr = na_tr/length(indices_anomalous) # training ratio
    vtr = (1 - tr)/2 # validation/test ratio
    split_inds_anomalous = train_val_test_inds(indices_anomalous, (tr, vtr, vtr); seed=seed)

    # this can be done universally - how?
    if nd == 1
        tr_n, val_n, tst_n = map(is -> data_normal[is], split_inds)
        tr_a, val_a, tst_a = map(is -> data_anomalous[is], split_inds_anomalous)
    elseif nd == 2
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
    load_data(dataset::String, ratios=(0.6,0.2,0.2); seed=nothing, 
    method="leave-one-out", contamination::Real=0.0, category, kwargs...)

Returns 3 tuples of (data, labels) representing train/validation/test part. Arguments are the splitting
ratios for normal data, seed and training data contamination.

For a list of available datasets, check `GenerativeAD.Datasets.uci_datasets`, `GenerativeAD.Datasets.other_datasets`,
`GenerativeAD.Datasets.mldatasets`. For MNIST-C and MVTec-AD datasets, the categories can be obtained by
 `GenerativeAD.Datasets.mnist_c_categories()` and `GenerativeAD.Datasets.mvtec_ad_categories()`. It can also
 be used to load the fixed `wildlife_MNIST` dataset.
"""
function load_data(dataset::String, ratios=(0.6,0.2,0.2); seed=nothing, 
    method="leave-one-out", contamination::Real=0.0, kwargs...)
    any(method .== ["leave-one-out","leave-one-in"]) ? nothing : error("unknown method, choose one of `leave-one-in`, `leave-one-out`")
    (method ==  "leave-one-in" && (!(dataset in mldatasets) & (dataset!="wildlife_MNIST"))) ? 
        error("`leave-one-in` only implemented for MNIST, FMNIST, SVHN2, CIFAR10 and wildlife_MNIST") : nothing

    # extract data and labels
    if dataset in uci_datasets # UCI Loda data, standardized
        data_normal, data_anomalous = load_uci_data(dataset; kwargs...)
    elseif dataset in mldatasets # MNIST,FMNIST, SVHN2, CIFAR10
        data_normal, data_anomalous = load_mldatasets_data(dataset; kwargs...)
    elseif dataset in other_datasets # other tabular datasets
        data_normal, data_anomalous = load_other_data(dataset; standardize=true, kwargs...)
    elseif dataset == "wildlife_MNIST"
        method == "leave-one-out" ? error("leave-one-out not implemented for wildlife_MNIST") : nothing
        if haskey(kwargs, :anomaly_class_ind)
            kwargs = merge(values(kwargs), (normal_class_ind=values(kwargs).anomaly_class_ind, anomaly_class_ind=-1))
        end
        (data_normal, y_normal), (data_anomalous, y_anomalous) = load_wildlife_mnist_data(;kwargs...)
    elseif dataset == "cocoplaces"
        method == "leave-one-out" ? error("leave-one-out not implemented for cocoplaces") : nothing
        if haskey(kwargs, :anomaly_class_ind)
            kwargs = merge(values(kwargs), (normal_class_ind=values(kwargs).anomaly_class_ind, anomaly_class_ind=-1))
        end
        (data_normal, y_normal), (data_anomalous, y_anomalous) = load_cocoplaces_data(;kwargs...)
    elseif dataset =="MNIST-C"
        data_normal, data_anomalous = load_mnist_c_data(; kwargs...)
    elseif dataset == "MVTec-AD"
        data_normal, data_anomalous = load_mvtec_ad_data(; kwargs...)
    elseif occursin("MNIST-C", dataset) # this is so one can pass both the dataset and category in one string
        category = dataset[9:end]
        data_normal, data_anomalous = load_mnist_c_data(; category=category, kwargs...)
    elseif occursin("MVTec-AD", dataset) 
        category = dataset[10:end]
        data_normal, data_anomalous = load_mvtec_ad_data(; category=category, kwargs...)
    else
        error("Dataset $(dataset) not known, either not implemented or misspeled.")
    end

    # now do the train/validation/test split
    if dataset == "wildlife_MNIST"
        return train_val_test_split(data_normal, data_anomalous, ratios; seed=seed, contamination=contamination)
    elseif method == "leave-one-in" # in this case, we swap the anomalous nad normal data
        return train_val_test_split(data_anomalous, data_normal, ratios; seed=seed, contamination=contamination)
    else
        return train_val_test_split(data_normal, data_anomalous, ratios; seed=seed, contamination=contamination)
    end
end

"""
    vectorize(data)

Vectorizes the 4D image data returned from `load_data`.
"""
vectorize(data) = map(d->(reshape(d[1], :, size(d[1],4)), d[2]),data)

function normalize_data(data)
    if minimum(data[1][1]) == 0 && maximum(data[1][1]) == 1
        return ((data[1][1] .- 0.5) ./0.5, data[1][2]), 
            ((data[2][1] .- 0.5) ./0.5, data[2][2]),
            ((data[3][1] .- 0.5) ./0.5, data[3][2])
    else
        return data
    end
end

"""
    split_multifactor_data(anomaly_factors, train_class, scores_orig, mf_scores, mf_labels; 
        mf_normal=false, seed=nothing)
"""
function split_multifactor_data(anomaly_factors, train_class, scores_orig, mf_scores, mf_labels; 
    mf_normal=false, seed=nothing)
    # get the original data - these are supposed to be clean
    val_scores_orig, tst_scores_orig = scores_orig

    # construct the normal and anomalous datasets
    ainds = map(i->mf_labels[i,:] .!= train_class, anomaly_factors)
    ainds = map(is -> reduce(|, is), zip(ainds...))
    a_scores, n_scores = if ndims(mf_scores) == 1
         mf_scores[ainds], mf_scores[.!ainds]
    elseif ndims(mf_scores) == 2
        mf_scores[:,ainds], mf_scores[:,.!ainds]
    else
        error("this only works for 1D and 2D arrays")
    end

    # now split the multifactor scores
    if mf_normal
        # this is much harder
        mf_split = train_val_test_split(n_scores, a_scores, (0.0, 0.5, 0.5), seed=1);
        val_scores, val_labels = mf_split[2]
        tst_scores, tst_labels = mf_split[3]
    else
        # this does not contain normal data from the mf dataset
        mf_split = (ndims(mf_scores) == 1) ? 
            train_val_test_split(n_scores[2:1], a_scores, (0.0, 0.5, 0.5), seed=1) :
            train_val_test_split(n_scores[:,2:1], a_scores, (0.0, 0.5, 0.5), seed=1);

        # get scores and labels for the evaluation function
        val_scores = cat(val_scores_orig, mf_split[2][1], dims=ndims(mf_scores));
        tst_scores = cat(tst_scores_orig, mf_split[3][1], dims=ndims(mf_scores));
        val_labels = vcat(zeros(size(val_scores_orig, ndims(val_scores_orig))), mf_split[2][2]);
        tst_labels = vcat(zeros(size(tst_scores_orig, ndims(tst_scores_orig))), mf_split[3][2]);
    end

    return (val_scores, val_labels), (tst_scores, tst_labels)
end
