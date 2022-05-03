#!/bin/bash

SOURCE=$1
TARGET=$2
FORCE=$3

LOG_DIR="${HOME}/logs/collect_stats"

if [ ! -d "$LOG_DIR" ]; then
    mkdir $LOG_DIR
fi

for i in {1..40} 
do
    # submit to slurm
    sbatch ./collect_stats_chunked.sh --output="${LOG_DIR}/%A.out" $i $SOURCE $TARGET $FORCE
done
