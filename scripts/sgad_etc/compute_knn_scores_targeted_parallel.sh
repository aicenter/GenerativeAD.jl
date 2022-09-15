#!/bin/bash

MODEL=$1
DATASET_FILE=$2
DATATYPE=$3
AC=$4
FORCE=$5

LOG_DIR="${HOME}/logs/knn_scores"

if [ ! -d "$LOG_DIR" ]; then
    mkdir $LOG_DIR
fi

while read d; do
    # submit to slurm
    sbatch \
    --output="${LOG_DIR}/${d}_${AC}_${MODEL}_%A.out" \
     ./compute_knn_scores_targeted.sh $MODEL $d $DATATYPE $AC $FORCE
done < ${DATASET_FILE}
