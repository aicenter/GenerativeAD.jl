#!/bin/bash
# this runs parallel experiments over all datasets
MODELSCRIPT=$1
MAX_SEED=$2
cat datasets_tabular.txt | xargs -n 1 -P 40 ./${MODELSCRIPT}.sh $MAX_SEED