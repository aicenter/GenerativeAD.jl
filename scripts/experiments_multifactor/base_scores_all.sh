#!/bin/bash
./base_scores_parallel.sh vae wildlife_MNIST -f
./base_scores_parallel.sh fmgan wildlife_MNIST -f
./base_scores_parallel.sh DeepSVDD wildlife_MNIST -f
./base_scores_parallel.sh fAnoGAN wildlife_MNIST -f
./base_scores_parallel.sh sgvae wildlife_MNIST -f
./base_scores_parallel.sh cgn wildlife_MNIST -f
