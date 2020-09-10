#!/bin/bash
#SBATCH --time=24:00:00
#SBATCH --nodes=1 --ntasks-per-node=1 --cpus-per-task=1
#SBATCH --mem=20G

MAX_SEED=$1
DATASET=$2

module load Julia/1.4.1-linux-x86_64
module load Python/3.8.2-GCCcore-9.3.0

# setup the library version 
source /home/skvarvit/bin/generativead-dev.sh

for ((SEED=1; SEED<=$MAX_SEED; SEED++))
do	
	julia ./knn.jl $SEED $DATASET
done 
