#!/bin/bash

LOG_DIR="${HOME}/logs/multifactor_collect_stats"

if [ ! -d "$LOG_DIR" ]; then
	mkdir $LOG_DIR
fi

# submit to slurm
for AC in {1..10}
do
    sbatch --output="${LOG_DIR}/%A_$AC.out" ./collect_stats.sh $AC
done
