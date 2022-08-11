#!/bin/bash
# this runs the computation for all datasets and score types at once

MODEL=$1
FORCE=$2

./compute_encodings_parallel.sh $MODEL ../experiments_images/datasets_images_color.txt leave-one-in $FORCE
./compute_encodings_parallel.sh $MODEL ../mvtec_ad/categories_sgvae.txt mvtec $FORCE
