#!/bin/bash
DATASET_TYPE=$1

if [ "$DATASET_TYPE" == "tabular" ]; then
    models=("aae" "abod" "gan" "if" "lof" "ocsvm" "pidforest" "vae" "vae_ocsvm" "adVAE" "GANomaly" "knn" "MAF" "RealNVP" "fmgan" "hbos" "loda" "MO_GAAL" "ocsvm_rbf" "sptn" "vae_knn" "wae" "DeepSVDD")

    for m in ${models[*]}
    do
        ./run_ensemble_parallel.sh ${m} tabular ../experiments_tabular/datasets_tabular.txt
    done
elif  [ "$DATASET_TYPE" == "images" ]; then
    models=("aae" "Conv-GANomaly" "DeepSVDD" "fAnoGAN-GP" "knn" "vae" "vae_ocsvm" "aae_ocsvm" "Conv-SkipGANomaly" "fAnoGAN" "fmgan" "ocsvm" "vae_knn" "wae")

    for m in ${models[*]}
    do
        ./run_ensemble_parallel.sh ${m} images ../experiments_images/datasets_images.txt
    done
else
    echo "Unsupported dataset type. Only 'images' or 'tabular' are supported."
fi
