#!/bin/bash

MODEL=$1
DATASET_FILE=$2
DATATYPE=$3
LATENT_SCORE=$4
ANOMALY_CLASS=$5
METHOD=$6
BASE_BETA=$7
FORCE=$8

LOG_DIR="${HOME}/logs/realpha_scores"

if [ ! -d "$LOG_DIR" ]; then
    mkdir $LOG_DIR
fi

while read d; do
    # submit to slurm
    sbatch \
    --output="${LOG_DIR}/${d}_${LATENT_SCORE}_%A.out" \
     ./recompute_alpha_scores_per_class.sh $MODEL $d $DATATYPE ${LATENT_SCORE} ${ANOMALY_CLASS} ${METHOD} ${BASE_BETA} $FORCE
done < ${DATASET_FILE}
