#!/bin/bash

DATASET_FILE=$1
DATATYPE=$2
FORCE=$3

LOG_DIR="${HOME}/logs/sgvae_encodings"

if [ ! -d "$LOG_DIR" ]; then
    mkdir $LOG_DIR
fi

while read d; do
    # submit to slurm
    sbatch \
    --output="${LOG_DIR}/${d}_%A.out" \
     ./compute_encodings.sh $d $DATATYPE $FORCE
done < ${DATASET_FILE}
