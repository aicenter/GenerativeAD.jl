#!/bin/bash

MODELFILE=$1
DATASET=$2
AF1=$3
AF2=$4
AF3=$5

LOG_DIR="${HOME}/logs/multifactor_experiment"

if [ ! -d "$LOG_DIR" ]; then
    mkdir $LOG_DIR
fi

while read MODELNAME; do
    # submit to slurm
    sbatch \
    --output="${LOG_DIR}/${MODELNAME}_${DATASET}_%A.out" \
     ./multifactor_experiment_mf.sh $MODELNAME $DATASET $AF1 $AF2 $AF3
done < ${MODELFILE}
