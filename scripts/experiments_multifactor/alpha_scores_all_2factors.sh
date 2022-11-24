#!/bin/bash

MODELNAME=$1
DATASET=$2
LATENT_SCORE=$3
METHOD=$4
BASE_BETA=$5

# one factor
./alpha_scores_parallel.sh $MODELNAME $DATASET ${LATENT_SCORE} $METHOD ${BASE_BETA} 1
./alpha_scores_parallel.sh $MODELNAME $DATASET ${LATENT_SCORE} $METHOD ${BASE_BETA} 2

# two factors
./alpha_scores_parallel.sh $MODELNAME $DATASET ${LATENT_SCORE} $METHOD ${BASE_BETA} 1 2
