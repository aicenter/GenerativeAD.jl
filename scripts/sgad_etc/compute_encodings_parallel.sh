#!/bin/bash

MODEL=$1
DATASET_FILE=$2
DATATYPE=$3
FORCE=$4

LOG_DIR="${HOME}/logs/alpha_encodings"

if [ ! -d "$LOG_DIR" ]; then
    mkdir $LOG_DIR
fi

while read d; do
    # submit to slurm
    sbatch \
    --output="${LOG_DIR}/${d}_%A.out" \
     ./compute_encodings.sh $MODEL $d $DATATYPE $FORCE
done < ${DATASET_FILE}
