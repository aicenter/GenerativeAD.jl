#!/bin/bash
DATASET=$1

./base_scores_parallel.sh vae $DATASET -f
./base_scores_parallel.sh DeepSVDD $DATASET -f
./base_scores_parallel.sh fAnoGAN $DATASET -f
./base_scores_parallel.sh sgvae $DATASET -f
./base_scores_parallel.sh cgn $DATASET -f

./base_scores_parallel.sh fmganpy $DATASET -f
./base_scores_parallel.sh vaegan $DATASET -f
./base_scores_parallel.sh sgvaegan $DATASET -f
./base_scores_parallel.sh sgvaegan10 $DATASET -f
./base_scores_parallel.sh vaegan10 $DATASET -f
./base_scores_parallel.sh fmganpy10 $DATASET -f
