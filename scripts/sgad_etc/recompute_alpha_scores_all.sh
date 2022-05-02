#!/bin/bash
FORCE=$1

./recompute_alpha_scores_parallel.sh ../experiments_images/datasets_images_color.txt leave-one-in knn $FORCE
./recompute_alpha_scores_parallel.sh ../experiments_images/datasets_images_color.txt leave-one-in normal $FORCE
./recompute_alpha_scores_parallel.sh ../experiments_images/datasets_images_color.txt leave-one-in kld $FORCE
./recompute_alpha_scores_parallel.sh ../experiments_images/datasets_images_color.txt leave-one-in normal_logpx $FORCE

./recompute_alpha_scores_parallel.sh ../mvtec_ad/categories_sgvae.txt mvtec knn $FORCE
./recompute_alpha_scores_parallel.sh ../mvtec_ad/categories_sgvae.txt mvtec normal $FORCE
./recompute_alpha_scores_parallel.sh ../mvtec_ad/categories_sgvae.txt mvtec kld $FORCE
./recompute_alpha_scores_parallel.sh ../mvtec_ad/categories_sgvae.txt mvtec normal_logpx $FORCE
