#!/bin/bash

MODELNAME=$1
DATASET=$2
LATENT_SCORE=$3
FORCE=$4

LOG_DIR="${HOME}/logs/multifactor_latent_scores"

if [ ! -d "$LOG_DIR" ]; then
    mkdir $LOG_DIR
fi

for AC in {1..10}; do
    # submit to slurm
    sbatch \
    --output="${LOG_DIR}/${MODELNAME}_${DATASET}_${AC}_%A.out" \
     ./latent_scores.sh $MODELNAME $DATASET ${LATENT_SCORE} $AC $FORCE
done
