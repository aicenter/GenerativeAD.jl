#!/bin/bash

DATASET_FILE=$1
DATATYPE=$2
LATENT_SCORE=$3
METHOD=$4
FORCE=$5

LOG_DIR="${HOME}/logs/sgvae_realpha_scores"

if [ ! -d "$LOG_DIR" ]; then
    mkdir $LOG_DIR
fi

while read d; do
    # submit to slurm
    sbatch \
    --output="${LOG_DIR}/${d}_${LATENT_SCORE}_%A.out" \
     ./recompute_alpha_scores.sh $d $DATATYPE ${LATENT_SCORE} ${P_NEGATIVE} ${METHOD} $FORCE
done < ${DATASET_FILE}
