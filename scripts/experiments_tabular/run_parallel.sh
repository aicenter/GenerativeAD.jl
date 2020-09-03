#!/bin/bash
# this runs parallel experiments over all datasets
MODEL=$1
MAX_SEED=$2
cat datasets_tabular.txt | xargs -n 1 -P 40 ./${MODEL}_run.sh $MAX_SEED