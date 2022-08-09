#!/bin/bash

DATASET_FILE=$1
NUM_SAMPLES=$2
NUM_CONC=$3

LOG_DIR="${HOME}/logs/supervised_classifier"

if [ ! -d "$LOG_DIR" ]; then
    mkdir $LOG_DIR
fi

while read d; do
    # submit to slurm
    sbatch \
    --array=1-${NUM_SAMPLES}%${NUM_CONC} \
    --output="${LOG_DIR}/${d}_%A_%a.out" \
     ./classifier.sh $d
done < ${DATASET_FILE}
