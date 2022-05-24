#!/bin/bash

MODELNAME=$1
DATASET=$2
LATENT_SCORE=$3

# one factor
./alpha_scores_parallel.sh $MODELNAME $DATASET ${LATENT_SCORE} logreg 1
./alpha_scores_parallel.sh $MODELNAME $DATASET ${LATENT_SCORE} logreg 2
./alpha_scores_parallel.sh $MODELNAME $DATASET ${LATENT_SCORE} logreg 3

# two factors
./alpha_scores_parallel.sh $MODELNAME $DATASET ${LATENT_SCORE} logreg 1 2
./alpha_scores_parallel.sh $MODELNAME $DATASET ${LATENT_SCORE} logreg 1 3
./alpha_scores_parallel.sh $MODELNAME $DATASET ${LATENT_SCORE} logreg 2 3

# three factors
./alpha_scores_parallel.sh $MODELNAME $DATASET ${LATENT_SCORE} logreg 1 2 3