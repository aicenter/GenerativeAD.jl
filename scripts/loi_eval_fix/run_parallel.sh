#!/bin/bash
# This runs parallel experiments over all datasets.
# USAGE EXAMPLE
# 	./run_parallel.sh vae datasets_tabular.txt
# Run from this folder only.
MODEL=$1 		# which model to run
DATASET_FILE=$2	# file with dataset list

LOG_DIR="${HOME}/logs/${MODEL}"

if [ ! -d "$LOG_DIR" ]; then
	mkdir $LOG_DIR
fi

while read d; do
	# submit to slurm
    sbatch \
    --output="${LOG_DIR}/${d}-%A_%a.out" \
     ./${MODEL}_run.sh $d

    # for local testing    
    # ./${MODEL}_run.sh $MAX_SEED $d 10
done < ${DATASET_FILE}
