#!/bin/bash

DATASET=$1
DATATYPE=$2
LATENT_SCORE=$3
DATASET_FILE=$4

LOG_DIR="${HOME}/logs/sgvae_latent_scores"

if [ ! -d "$LOG_DIR" ]; then
    mkdir $LOG_DIR
fi

while read d; do
    # submit to slurm
    sbatch \
    --output="${LOG_DIR}/${d}.out" \
     ./compute_latent_scores.sh $d $DATATYPE ${LATENT_SCORE}
done < ${DATASET_FILE}
