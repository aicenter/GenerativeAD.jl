#!/bin/bash
# this runs the computation for all datasets and score types at once
FORCE=$1

./compute_encodings_parallel.sh ../experiments_images/datasets_images_color.txt leave-one-in $FORCE
./compute_encodings_parallel.sh ../mvtec_ad/categories_sgvae.txt mvtec $FORCE
