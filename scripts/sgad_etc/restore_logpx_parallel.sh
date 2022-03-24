#!/bin/bash
TYPE=$1        # mvtec or classic
DATASET_FILE=$2

LOG_DIR="${HOME}/logs/log_px_${TYPE}"

if [ ! -d "$LOG_DIR" ]; then
    mkdir $LOG_DIR
fi

while read d; do
    # submit to slurm
    sbatch \
    --output="${LOG_DIR}/${TYPE}_${d}.out" \
     ./restore_logpx.sh $TYPE $d
done < ${DATASET_FILE}
