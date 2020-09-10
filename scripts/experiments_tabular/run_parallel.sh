#!/bin/bash
# this runs parallel experiments over all datasets
MODEL=$1 		# which model to run
NUM_SAMPLES=$2	# how many repetitions
MAX_SEED=$3		# how many folds over dataset
NUM_CONC=$4		# number of concurrent tasks in the array job

while read d; do
    sbatch --array=1-${NUM_SAMPLES}%${NUM_CONC} ./${MODEL}_run.sh $MAX_SEED $d
    # ./${MODEL}_run.sh $MAX_SEED $d
done < datasets_tabular.txt