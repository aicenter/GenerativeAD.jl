using DrWatson
@quickactivate
using GenerativeAD
using NPZ
import GenerativeAD.Datasets: load_data, load_wildlife_mnist_data, train_val_test_inds

#
outdir = datadir("wildlife_MNIST/training_splits")
mkpath(outdir)

function get_class_labels(ac, seed)
    (data_normal, y_normal), (data_anomalous, y_anomalous) = load_wildlife_mnist_data(normal_class_ind=ac);
    ratios = (0.6,0.2,0.2)
    contamination = 0.0

    # this is taken from "train_val_test_split"
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

    tr_n, val_n, tst_n = map(is -> y_normal[is], split_inds)
    tr_a, val_a, tst_a = map(is -> y_anomalous[is], split_inds_anomalous)

    # cat it together
    tr_c = cat(tr_n, tr_a, dims = 1)
    val_c = cat(val_n, val_a, dims = 1)
    tst_c = cat(tst_n, tst_a, dims = 1)

    tr_c, val_c, tst_c
end

# load data
dataset = "wildlife_MNIST"
for seed in 1:1
    for ac in 1:10
        # load the data and also the anomaly labels
        (tr_x, tr_y), (val_x, val_y), (tst_x, tst_y) = load_data(dataset, seed=seed, anomaly_class_ind=ac, 
            method="leave-one-in");
        tr_x = permutedims(tr_x, (4,3,2,1))
        val_x = permutedims(val_x, (4,3,2,1))
        tst_x = permutedims(tst_x, (4,3,2,1))

        # get the class labels
        tr_c, val_c, tst_c = get_class_labels(ac, seed);

        # save data
        npzwrite(joinpath(outdir, "ac=$(ac)_seed=$(seed)_train_data.npy"), tr_x)
        npzwrite(joinpath(outdir, "ac=$(ac)_seed=$(seed)_train_labels.npy"), tr_y)
        npzwrite(joinpath(outdir, "ac=$(ac)_seed=$(seed)_train_classes.npy"), tr_c)

        npzwrite(joinpath(outdir, "ac=$(ac)_seed=$(seed)_validation_data.npy"), val_x)
        npzwrite(joinpath(outdir, "ac=$(ac)_seed=$(seed)_validation_labels.npy"), val_y)
        npzwrite(joinpath(outdir, "ac=$(ac)_seed=$(seed)_validation_classes.npy"), val_c)

        npzwrite(joinpath(outdir, "ac=$(ac)_seed=$(seed)_test_data.npy"), tst_x)
        npzwrite(joinpath(outdir, "ac=$(ac)_seed=$(seed)_test_labels.npy"), tst_y)
        npzwrite(joinpath(outdir, "ac=$(ac)_seed=$(seed)_test_classes.npy"), tst_c)
    end
end
