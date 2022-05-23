#!/bin/bash

FORCE=$1

./latent_scores_parallel.sh sgvae wildlife_MNIST normal $FORCE
./latent_scores_parallel.sh sgvae wildlife_MNIST normal_logpx $FORCE
./latent_scores_parallel.sh sgvae wildlife_MNIST kld $FORCE