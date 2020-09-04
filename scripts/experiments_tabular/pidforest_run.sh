#!/bin/bash
# example: `./pidforest_run.sh iris 1`
MAX_SEED=$1
DATASET=$2

for ((SEED=1; SEED<=$MAX_SEED; SEED++))
do	
	for ITER in {1..100}
	do
		julia pidforest.jl $SEED $DATASET
	done
done