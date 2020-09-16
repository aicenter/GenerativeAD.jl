#!/bin/bash
#SBATCH --time=24:00:00
#SBATCH --nodes=1 --ntasks-per-node=4 --cpus-per-task=1
#SBATCH --mem=16G
#SBATCH --qos==collaborator

MAX_SEED=$1
DATASET=$2

module load Julia/1.4.1-linux-x86_64

julia --project ./real_nvp.jl $MAX_SEED $DATASET
