#!/bin/bash

SOURCE=$1
TARGET=$2
FORCE=$3

LOG_DIR="${HOME}/logs/collect_stats"

if [ ! -d "$LOG_DIR" ]; then
    mkdir $LOG_DIR
fi

for i in {1..50} 
do
    # submit to slurm
    sbatch --output="${LOG_DIR}/%A.out" ./collect_stats_chunked.sh $i $SOURCE $TARGET $FORCE
done
