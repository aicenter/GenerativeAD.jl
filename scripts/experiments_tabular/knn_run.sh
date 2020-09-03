#!/bin/bash
# example: `./knn_run.sh iris 1`
MAX_SEED=$1
DATASET=$2
# do a parallel loop here
for ((SEED=1; SEED<=$MAX_SEED; SEED++))
do	
	for ITER in {1..100}
	do
		julia knn.jl $SEED $DATASET
	done
done