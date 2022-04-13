#!/bin/bash

DATASET_FILE=$1
DATATYPE=$2
AC=$3
FORCE=$4

LOG_DIR="${HOME}/logs/sgvae_knn_scores"

if [ ! -d "$LOG_DIR" ]; then
    mkdir $LOG_DIR
fi

while read d; do
    # submit to slurm
    sbatch \
    --output="${LOG_DIR}/${d}_${AC}_%A.out" \
     ./compute_knn_scores_per_class.sh $d $DATATYPE $AC $FORCE
done < ${DATASET_FILE}
