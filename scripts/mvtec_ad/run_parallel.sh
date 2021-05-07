#!/bin/bash
# This runs parallel experiments over all datasets.
# USAGE EXAMPLE
# 	./run_parallel.sh pidforest 3 1 2 datasets_tabular.txt 0.01
# Run from this folder only.
MODEL=$1 		# which model to run
CATEGORY=$2	    # which mvtec ad category
NUM_SAMPLES=$3	# how many repetitions
NUM_CONC=$4		# number of concurrent tasks in the array job
SEED_FILE=$5		# file with the individual seed number
CONTAMINATION=$6

LOG_DIR="${HOME}/logs/${MODEL}"

if [ ! -d "$LOG_DIR" ]; then
	mkdir $LOG_DIR
fi

while read SEED; do
	# submit to slurm
    sbatch \
    --array=1-${NUM_SAMPLES}%${NUM_CONC} \
    --output="${LOG_DIR}/MVTEC_${CATEGORY}-SEED_${SEED}-%A_%a.out" \
     ./${MODEL}_run.sh ${SEED} $CATEGORY $CONTAMINATION

    # for local testing    
    # ./${MODEL}_run.sh $MAX_SEED $d 10
done < ${SEED_FILE}
