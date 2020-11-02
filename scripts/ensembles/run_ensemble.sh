#!/bin/bash
#SBATCH --time=24:00:00
#SBATCH --nodes=1 --ntasks-per-node=2 --cpus-per-task=1
#SBATCH --mem=24G

MODEL=$1 		# which model to run
DATASET=$2      # which dataset to run
DATASET_TYPE=$3	# images | tabular

julia --project ./ensemble.jl $MODEL $DATASET $DATASET_TYPE 5 10