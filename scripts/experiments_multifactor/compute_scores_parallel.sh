#!/bin/bash

MODELNAME=$1
DATASET_FILE=$2
FORCE=$3

LOG_DIR="${HOME}/logs/experiments_multifactor"

if [ ! -d "$LOG_DIR" ]; then
    mkdir $LOG_DIR
fi

while read d; do
    # submit to slurm
    sbatch \
    --output="${LOG_DIR}/${MODELNAME}_${d}_%A.out" \
     ./compute_scores.sh $MODELNAME $d $FORCE
done < ${DATASET_FILE}
