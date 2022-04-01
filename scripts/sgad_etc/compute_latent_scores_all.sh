#!/bin/bash
# this runs the computation for all datasets and score types at once
./compute_latent_scores_parallel.sh ../experiments_images/datasets_images_color.txt leave-one-in normal
./compute_latent_scores_parallel.sh ../experiments_images/datasets_images_color.txt leave-one-in kld
./compute_latent_scores_parallel.sh ../experiments_images/datasets_images_color.txt leave-one-in normal_logpx

./compute_latent_scores_parallel.sh ../mvtec_ad/categories_cgn.txt mvtec normal
./compute_latent_scores_parallel.sh ../mvtec_ad/categories_cgn.txt mvtec kld
./compute_latent_scores_parallel.sh ../mvtec_ad/categories_cgn.txt mvtec normal_logpx

./compute_latent_scores_parallel.sh ../mvtec_ad/categories_mvtec.txt mvtec normal
./compute_latent_scores_parallel.sh ../mvtec_ad/categories_mvtec.txt mvtec kld
./compute_latent_scores_parallel.sh ../mvtec_ad/categories_mvtec.txt mvtec normal_logpx
