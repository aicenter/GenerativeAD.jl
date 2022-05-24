#!/bin/bash

MODELNAME=$1
DATASET=$2
LATENT_SCORE=$3
METHOD=$4
AF1=$5
AF2=$6
AF3=$7


LOG_DIR="${HOME}/logs/alpha_scores_multifactor"

if [ ! -d "$LOG_DIR" ]; then
    mkdir $LOG_DIR
fi

for AC in {1..10}; do
    # submit to slurm
    sbatch \
    --output="${LOG_DIR}/${MODELNAME}_${DATASET}_${AC}_%A.out" \
     ./alpha_scores.sh $MODELNAME $DATASET ${LATENT_SCORE} $AC $METHOD $AF1 $AF2 $AF3
done
