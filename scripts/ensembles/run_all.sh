#!/bin/bash
DATASET_TYPE=$1

if [ "$DATASET_TYPE" == "tabular" ]; then
    models=("aae_full" "wae_full" "vae_full" "adVAE" "GANomaly" "abod" "hbos" "loda" "if" "lof" "ocsvm" "pidforest" "knn" "MAF" "RealNVP" "sptn" "gan" "fmgan" "MO_GAAL" "vae_knn" "vae_ocsvm" "repen" "dagmm" "DeepSVDD")

    for m in ${models[*]}
    do
        ./run_ensemble_parallel.sh ${m} tabular ../experiments_tabular/datasets_tabular.txt
    done
elif  [ "$DATASET_TYPE" == "images_leave-one-out" ] || [ "$DATASETS" == "images_leave-one-in" ] ; then
    models=("aae" "Conv-GANomaly" "Conv-SkipGANomaly" "vae" "wae" "knn" "ocsvm" "fAnoGAN" "fmgan" "DeepSVDD" "vae_ocsvm" "vae_knn")

    for m in ${models[*]}
    do
        ./run_ensemble_parallel.sh ${m} ${DATASET_TYPE} ../experiments_images/datasets_images.txt
    done
elif  [ "$DATASET_TYPE" == "images_mnistc" ]; then
    models=("aae" "Conv-GANomaly" "Conv-SkipGANomaly" "vae" "wae" "knn" "ocsvm" "fAnoGAN" "fmgan" "DeepSVDD" "vae_ocsvm" "vae_knn")

    for m in ${models[*]}
    do
        ./run_ensemble_parallel.sh ${m} ${DATASET_TYPE} ../experiments_images/datasets_mnistc.txt
    done
elif  [ "$DATASET_TYPE" == "images_mvtec" ]; then
    models=("aae" "Conv-GANomaly" "Conv-SkipGANomaly" "vae" "wae" "knn" "ocsvm" "fAnoGAN" "fmgan" "DeepSVDD" "vae_ocsvm" "vae_knn")

    for m in ${models[*]}
    do
        ./run_ensemble_parallel.sh ${m} ${DATASET_TYPE} ../mvtec_ad/categories_mvtec.txt
    done
else
    echo "Unsupported dataset type. Only 'tabular, 'images_leave-one-out/in', 'images_mnistc' or 'images_mvtec' are supported."
fi
