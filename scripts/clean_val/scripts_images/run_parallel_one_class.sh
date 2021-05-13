#!/bin/bash
# This runs parallel experiments over all datasets.
# USAGE EXAMPLE
# 	./run_parallel.sh knn 3 1 1 datasets_iamge.txt 5 leave-one-out 0.01
# Run from this folder only.
MODEL=$1 		# which model to run
NUM_SAMPLES=$2	# how many repetitions
MAX_SEED=$3		# how many folds over dataset
NUM_CONC=$4		# number of concurrent tasks in the array job
DATASET_FILE=$5	# file with dataset list
ANOMALY_CLASS=$6
METHOD=$7
CONTAMINATION=$8

LOG_DIR="${HOME}/logs/${MODEL}"

if [ ! -d "$LOG_DIR" ]; then
	mkdir $LOG_DIR
fi

while read d; do
	# submit to slurm
    sbatch \
    --array=1-${NUM_SAMPLES}%${NUM_CONC} \
    --output="${LOG_DIR}/${d}-%A_%a.out" \
     ./${MODEL}_run.sh $MAX_SEED $d ${ANOMALY_CLASS} $METHOD $CONTAMINATION

    # for local testing    
    # ./${MODEL}_run.sh $MAX_SEED $d 10
done < ${DATASET_FILE}
