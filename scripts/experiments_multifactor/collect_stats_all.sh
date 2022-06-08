#!/bin/bash

LOG_DIR="${HOME}/logs/multifactor_collect_stats"

if [ ! -d "$LOG_DIR" ]; then
	mkdir $LOG_DIR
fi

# submit to slurm
sbatch \
--array=1-10 \
--output="${LOG_DIR}/%A_%a.out" \
 ./collect_stats.sh %a
