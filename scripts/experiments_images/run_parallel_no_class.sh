#!/bin/bash
# This runs parallel experiments over all datasets
# a version for image datasets without multiple anomaly classes (MVTec-AD, MNIST-C)
# USAGE EXAMPLE
# 	./run_parallel_no_class.sh vae 3 1 2 datasets_mnistc.txt 0.1
# Run from this folder only.
MODEL=$1 		# which model to run
NUM_SAMPLES=$2	# how many repetitions
MAX_SEED=$3		# how many folds over dataset
NUM_CONC=$4		# number of concurrent tasks in the array job
DATASET_FILE=$5	# file with dataset list
CONTAMINATION=$6

LOG_DIR="${HOME}/logs/${MODEL}"

if [ ! -d "$LOG_DIR" ]; then
	mkdir $LOG_DIR
fi

while read d; do
	# submit to slurm
    sbatch \
    --array=1-${NUM_SAMPLES}%${NUM_CONC} \
    --output="${LOG_DIR}/${d}-%A_%a.out" \
     ./${MODEL}_run.sh $MAX_SEED $d 1 "leave-one-out" $CONTAMINATION

    # for local testing    
    # ./${MODEL}_run.sh $MAX_SEED $d 10
done < ${DATASET_FILE}
