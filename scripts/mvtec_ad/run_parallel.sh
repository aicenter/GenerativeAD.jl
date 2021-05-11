#!/bin/bash
# This runs parallel experiments over all datasets.
# USAGE EXAMPLE
# 	./run_parallel.sh vae 20 5 10 categories_mvtec.txt 0.01
# Run from this folder only.
MODEL=$1 		 # which model to run
NUM_SAMPLES=$2	 # how many repetitions
MAX_SEED=$3		 # max seed
NUM_CONC=$4		 # number of concurrent tasks in the array job
CATEGORY_FILE=$5 # file with categories to be iterated over
CONTAMINATION=$6

LOG_DIR="${HOME}/logs/${MODEL}"

if [ ! -d "$LOG_DIR" ]; then
	mkdir $LOG_DIR
fi

while read CATEGORY; do
	# submit to slurm
    sbatch \
    --array=1-${NUM_SAMPLES}%${NUM_CONC} \
    --output="${LOG_DIR}/mvtec_${CATEGORY}-%A_%a.out" \
     ./${MODEL}_run.sh ${MAX_SEED} $CATEGORY $CONTAMINATION

    # for local testing    
    # ./${MODEL}_run.sh $MAX_SEED $d 10
done < ${CATEGORY_FILE}
