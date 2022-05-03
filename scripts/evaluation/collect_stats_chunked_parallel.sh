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
    sbatch \
    --output="${LOG_DIR}/${SOURCE}_%A.out" \
     ./collect_stats_chunked.sh $i $SOURCE $TARGET $FORCE
done

LOG_DIR="${HOME}/logs/collect_stats"
i=1
SOURCE="evaluation_kp/images_mvtec"
TARGET="evaluation_kp/images_mvtec_cache"
FORCE="-f"

sbatch --output="${LOG_DIR}/${SOURCE}_%A.out" ./collect_stats_chunked.sh $i $SOURCE $TARGET $FORCE


./collect_stats_chunked_parallel.sh evaluation_kp/images_mvtec evaluation_kp/images_mvtec_cache -f