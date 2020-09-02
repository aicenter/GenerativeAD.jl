#!/bin/bash
# example: `./knn_run.sh iris 1`
DATASET=$1
SEED=$2
# do a parallel loop here
for iter in {1..100}
do
	julia knn.jl $DATASET $SEED 
done