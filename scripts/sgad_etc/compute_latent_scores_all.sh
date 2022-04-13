#!/bin/bash
# this runs the computation for all datasets and score types at once
FORCE=$1

./compute_latent_scores_parallel.sh ../experiments_images/datasets_images_color.txt leave-one-in normal $FORCE
./compute_latent_scores_parallel.sh ../experiments_images/datasets_images_color.txt leave-one-in kld $FORCE
./compute_latent_scores_parallel.sh ../experiments_images/datasets_images_color.txt leave-one-in normal_logpx $FORCE

./compute_latent_scores_parallel.sh ../mvtec_ad/categories_sgvae.txt mvtec normal $FORCE
./compute_latent_scores_parallel.sh ../mvtec_ad/categories_sgvae.txt mvtec kld $FORCE
./compute_latent_scores_parallel.sh ../mvtec_ad/categories_sgvae.txt mvtec normal_logpx $FORCE
