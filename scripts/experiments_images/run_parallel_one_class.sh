#!/bin/bash
# This runs parallel experiments over all datasets, but just for a single anomaly class.
# USAGE EXAMPLE
# ./run_parallel_one_class.sh knn 10 1 10 dataset_svhn.txt 1 0.01
# Run from this folder only.
MODEL=$1 		# which model to run
NUM_SAMPLES=$2	# how many repetitions
MAX_SEED=$3		# how many folds over dataset
NUM_CONC=$4		# number of concurrent tasks in the array job
DATASET_FILE=$5	# file with dataset list
ANOMALY_CLASS=$6 # which anomaly class to run
CONTAMINATION=$7 # training contamination

LOG_DIR="${HOME}/logs/${MODEL}"

if [ ! -d "$LOG_DIR" ]; then
	mkdir $LOG_DIR
fi

while read d; do
	# submit to slurm
    sbatch \
    --array=1-${NUM_SAMPLES}%${NUM_CONC} \
    --output="${LOG_DIR}/${d}-%A_%a.out" \
     ./${MODEL}_one_class_run.sh $MAX_SEED $d ${ANOMALY_CLASS} $CONTAMINATION

    # for local testing    
    # ./${MODEL}_run.sh $MAX_SEED $d 10
done < ${DATASET_FILE}
