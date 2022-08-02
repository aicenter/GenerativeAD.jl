#!/bin/bash

DATASET_FILE=$1
DATATYPE=$2
LATENT_SCORE=$3
ANOMALY_CLASS=$4
METHOD=$5
BASE_BETA=$6
FORCE=$7

LOG_DIR="${HOME}/logs/sgvae_realpha_scores"

if [ ! -d "$LOG_DIR" ]; then
    mkdir $LOG_DIR
fi

while read d; do
    # submit to slurm
    sbatch \
    --output="${LOG_DIR}/${d}_${LATENT_SCORE}_%A.out" \
     ./recompute_alpha_scores_per_class.sh $d $DATATYPE ${LATENT_SCORE} ${ANOMALY_CLASS} ${METHOD} ${BASE_BETA} $FORCE
done < ${DATASET_FILE}
