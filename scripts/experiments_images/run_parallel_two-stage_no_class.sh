#!/bin/bash
# This runs parallel experiments over all datasets.
# USAGE EXAMPLE
#       ./run_parallel_two-stage_no_class.sh two-stage_maf 10 5 10 datasets_mnistc.txt vae_LOSS_images_mnistc 1
# Run from this folder only.
MODEL=$1                # which model to run
NUM_SAMPLES=$2  # how many repetitions
MAX_SEED=$3             # how many folds over dataset
NUM_CONC=$4             # number of concurrent tasks in the array job
DATASET_FILE=$5 # file with dataset list
TAB_NAME=$6
MI_ONLY=$7
CONTAMINATION=$8

LOG_DIR="${HOME}/logs/${MODEL}-${TAB_NAME}"

if [ ! -d "$LOG_DIR" ]; then
        mkdir $LOG_DIR
fi

while read d; do
        # submit to slurm
    sbatch \
    --array=1-${NUM_SAMPLES}%${NUM_CONC} \
    --output="${LOG_DIR}/${d}-%A_%a.out" \
     ./two-stage_${MODEL}_run.sh ${MAX_SEED} $d 1 $MI_ONLY ${TAB_NAME} "leave-one-out" $CONTAMINATION

    # for local testing    
    # ./${MODEL}_run.sh $MAX_SEED $d 10
done < ${DATASET_FILE}