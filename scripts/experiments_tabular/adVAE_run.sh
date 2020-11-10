#!/bin/bash
#SBATCH --time=24:00:00
#SBATCH --nodes=1 --ntasks-per-node=2 --cpus-per-task=1
#SBATCH --mem=40G

MAX_SEED=$1
DATASET=$2

module load Julia/1.5.1-linux-x86_64
module load Python/3.8.2-GCCcore-9.3.0

julia ./adVAE.jl ${MAX_SEED} $DATASET 
