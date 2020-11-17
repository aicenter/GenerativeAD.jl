#!/bin/bash
DATASET_TYPE=$1

if [ "$DATASET_TYPE" == "tabular" ]; then
    models=("GANomaly" "MAF" "MO_GAAL" "RealNVP" "aae" "aae_vamp" "abod" "adVAE" "fmgan" "gan" "hbos" "if" "knn" "loda" "lof" "ocsvm_nu" "ocsvm_rbf" "ocsvm" "pidforest" "sptn" "vae" "wae" "wae_vamp")

    for m in ${models[*]}
    do
        ./run_ensemble_parallel.sh ${m} tabular ../experiments_tabular/datasets_tabular.txt
    done
elif  [ "$DATASET_TYPE" == "images" ]; then
    models=("Conv-GANomaly" "Conv-SkipGANomaly" "DeepSVDD" "aae" "fmgan" "knn" "ocsvm" "vae" "wae")

    for m in ${models[*]}
    do
        ./run_ensemble_parallel.sh ${m} images ../experiments_images/datasets_images.txt
    done
else
    echo "Unsupported dataset type. Only 'images' or 'tabular' are supported."
fi
