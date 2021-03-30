#!/bin/bash
# This runs parallel experiments over all datasets.
# USAGE EXAMPLE
# 	./run_parallel.sh pidforest 3 1 2 datasets_tabular.txt random 0.05
# Run from this folder only.
MODEL=$1 		 # which model to run
NUM_SAMPLES=$2	 # how many repetitions
MAX_SEED=$3		 # how many folds over dataset
NUM_CONC=$4		 # number of concurrent tasks in the array job
DATASET_FILE=$5	 # file with dataset list
HP_SAMPLING=$6   # method for sampling hyperparameters [random|bayes]
CONTAMINATION=$7 # training data contamination

LOG_DIR="${HOME}/logs/${MODEL}"

if [ ! -d "$LOG_DIR" ]; then
	mkdir $LOG_DIR
fi

if [ "$HP_SAMPLING" == "bayes" ] && [ $NUM_CONC > 1] then
    echo "Bayesian hyperparameter optimization allows only 1 concurent job, ${NUM_CONC} provided."
else
    while read d; do
        # submit to slurm
        sbatch \
        --array=1-${NUM_SAMPLES}%${NUM_CONC} \
        --output="${LOG_DIR}/${d}-%A_%a.out" \
        ./${MODEL}_run.sh $MAX_SEED $d $HP_SAMPLING $CONTAMINATION

        # for local testing    
        # ./${MODEL}_run.sh $MAX_SEED $d
    done < ${DATASET_FILE}
fi