#!/bin/bash
P_NEGATIVE=$1
VALAUC=$2

./gather_alpha_scores_parallel.sh ../experiments_images/datasets_images_color.txt leave-one-in 10 ${P_NEGATIVE} $VALAUC
./gather_alpha_scores_parallel.sh ../mvtec_ad/categories_sgvae.txt mvtec 1 ${P_NEGATIVE} $VALAUC
