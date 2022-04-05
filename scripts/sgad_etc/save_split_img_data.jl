using DrWatson
@quickactivate
using GenerativeAD
using NPZ
import GenerativeAD.Datasets: load_data, load_wildlife_mnist_data, train_val_test_inds

#
dataset = "CIFAR10"
dataset = "SVHN2"
outdir = datadir("$(dataset)/training_splits")
mkpath(outdir)

# load data
for seed in 1:1
    for ac in 1:10
        # load the data and also the anomaly labels
        (tr_x, tr_y), (val_x, val_y), (tst_x, tst_y) = load_data(dataset, seed=seed, anomaly_class_ind=ac, 
            method="leave-one-in");
        tr_x = permutedims(tr_x, (4,3,2,1))
        val_x = permutedims(val_x, (4,3,2,1))
        tst_x = permutedims(tst_x, (4,3,2,1))

        # save data
        npzwrite(joinpath(outdir, "ac=$(ac)_seed=$(seed)_train_data.npy"), tr_x)
        npzwrite(joinpath(outdir, "ac=$(ac)_seed=$(seed)_train_labels.npy"), tr_y)

        npzwrite(joinpath(outdir, "ac=$(ac)_seed=$(seed)_validation_data.npy"), val_x)
        npzwrite(joinpath(outdir, "ac=$(ac)_seed=$(seed)_validation_labels.npy"), val_y)

        npzwrite(joinpath(outdir, "ac=$(ac)_seed=$(seed)_test_data.npy"), tst_x)
        npzwrite(joinpath(outdir, "ac=$(ac)_seed=$(seed)_test_labels.npy"), tst_y)
    end
end
