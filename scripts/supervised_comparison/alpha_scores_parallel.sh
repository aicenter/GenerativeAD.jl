#!/bin/bash

DATASET_FILE=$1
LATENT_SCORE=$2
ANOMALY_CLASS=$3
BASE_BETA=$4
FORCE=$5

LOG_DIR="${HOME}/logs/supervised_alpha_scores"

if [ ! -d "$LOG_DIR" ]; then
    mkdir $LOG_DIR
fi

while read d; do
    # submit to slurm
    sbatch \
    --output="${LOG_DIR}/${d}_${LATENT_SCORE}_%A.out" \
     ./alpha_scores.sh $d ${LATENT_SCORE} ${ANOMALY_CLASS} ${BASE_BETA} $FORCE
done < ${DATASET_FILE}
