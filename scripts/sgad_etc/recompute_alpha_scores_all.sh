#!/bin/bash
METHOD=$1
FORCE=$2

./recompute_alpha_scores_parallel.sh ../experiments_images/datasets_images_color.txt leave-one-in normal ${METHOD} $FORCE
./recompute_alpha_scores_parallel.sh ../experiments_images/datasets_images_color.txt leave-one-in kld ${METHOD} $FORCE
./recompute_alpha_scores_parallel.sh ../experiments_images/datasets_images_color.txt leave-one-in normal_logpx ${METHOD} $FORCE

./recompute_alpha_scores_parallel.sh ../mvtec_ad/categories_sgvae.txt mvtec normal ${METHOD} $FORCE
./recompute_alpha_scores_parallel.sh ../mvtec_ad/categories_sgvae.txt mvtec kld ${METHOD} $FORCE
./recompute_alpha_scores_parallel.sh ../mvtec_ad/categories_sgvae.txt mvtec normal_logpx ${METHOD} $FORCE
