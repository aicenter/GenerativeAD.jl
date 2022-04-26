#!/bin/bash
VALAUC=$1
./gather_alpha_scores_parallel.sh ../experiments_images/datasets_images_color.txt leave-one-in 10 $VALAUC
./gather_alpha_scores_parallel.sh ../mvtec_ad/categories_sgvae.txt mvtec 1 $VALAUC
