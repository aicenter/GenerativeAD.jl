#!/bin/bash
./regather_alpha_scores_parallel.sh ../experiments_images/datasets_images_color.txt leave-one-in 10
./regather_alpha_scores_parallel.sh ../mvtec_ad/categories_sgvae.txt mvtec 1
