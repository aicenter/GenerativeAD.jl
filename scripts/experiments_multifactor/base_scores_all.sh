#!/bin/bash
DATASET=$1

./base_scores_parallel.sh vae $DATASET -f
./base_scores_parallel.sh fmgan $DATASET -f
./base_scores_parallel.sh DeepSVDD $DATASET -f
./base_scores_parallel.sh fAnoGAN $DATASET -f
./base_scores_parallel.sh sgvae $DATASET -f
./base_scores_parallel.sh cgn $DATASET -f
