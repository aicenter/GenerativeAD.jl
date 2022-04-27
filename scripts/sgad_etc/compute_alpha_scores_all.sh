#!/bin/bash
P_NEGATIVE=$1
FORCE=$2

./compute_alpha_scores_parallel.sh ../experiments_images/datasets_images_color.txt leave-one-in knn ${P_NEGATIVE} $FORCE
./compute_alpha_scores_parallel.sh ../experiments_images/datasets_images_color.txt leave-one-in normal ${P_NEGATIVE} $FORCE
./compute_alpha_scores_parallel.sh ../experiments_images/datasets_images_color.txt leave-one-in kld ${P_NEGATIVE} $FORCE
./compute_alpha_scores_parallel.sh ../experiments_images/datasets_images_color.txt leave-one-in normal_logpx ${P_NEGATIVE} $FORCE

./compute_alpha_scores_parallel.sh ../mvtec_ad/categories_sgvae.txt mvtec knn ${P_NEGATIVE} $FORCE
./compute_alpha_scores_parallel.sh ../mvtec_ad/categories_sgvae.txt mvtec normal ${P_NEGATIVE} $FORCE
./compute_alpha_scores_parallel.sh ../mvtec_ad/categories_sgvae.txt mvtec kld ${P_NEGATIVE} $FORCE
./compute_alpha_scores_parallel.sh ../mvtec_ad/categories_sgvae.txt mvtec normal_logpx ${P_NEGATIVE} $FORCE
