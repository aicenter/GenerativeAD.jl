#!/bin/bash

FORCE=$1

LOG_DIR="${HOME}/logs/sgvae_regenerate_stats"

if [ ! -d "$LOG_DIR" ]; then
    mkdir $LOG_DIR
fi

for MODEL in cgn sgvae
do
    # submit to slurm
    sbatch \
    --output="${LOG_DIR}/${MODEL}_%A.out" \
     ./regenerate_stats_new.sh $MODEL $FORCE
done
