#!/bin/bash
# this runs parallel experiments over all datasets
MODEL=$1
NUM_SAMPLES=$2
MAX_SEED=$3

while read d; do
    sbatch --array=1-$NUM_SAMPLES -D /home/francja5/Projects/modules/GenerativeAD/scripts/experiments_tabular/ ./${MODEL}_run.sh $MAX_SEED $d
    # ./${MODEL}_run.sh $MAX_SEED $d
done < datasets_tabular.txt