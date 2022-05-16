#!/bin/bash

MODELNAME=$1
DATASET=$2
FORCE=$3

LOG_DIR="${HOME}/logs/experiments_multifactor"

if [ ! -d "$LOG_DIR" ]; then
    mkdir $LOG_DIR
fi

for AC in {1..10}; do
    # submit to slurm
    sbatch \
    --output="${LOG_DIR}/${MODELNAME}_${AC}_%A.out" \
     ./compute_scores.sh $MODELNAME $DATASET $AC $FORCE
done
