#!/bin/bash

DATASET_FILE=$1
DATATYPE=$2
MAX_AC=$3

LOG_DIR="${HOME}/logs/sgvae_gather_alpha_scores"

if [ ! -d "$LOG_DIR" ]; then
    mkdir $LOG_DIR
fi

for AC in {1..${MAX_AC}}
do
    while read d; do
        # submit to slurm
        sbatch \
        --output="${LOG_DIR}/${d}_${LATENT_SCORE}_${AC}_%A.out" \
         ./gather_alpha_scores.sh $d $DATATYPE $AC
    done < ${DATASET_FILE}
done