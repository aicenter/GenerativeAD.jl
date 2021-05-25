#!/bin/bash
#SBATCH --time=24:00:00
#SBATCH --nodes=1 --ntasks-per-node=2 --cpus-per-task=2
#SBATCH --gres=gpu:1
#SBATCH --partition=gpu
#SBATCH --mem=80G

DATASET=$1
CLASS=$2

module load Julia/1.5.1-linux-x86_64
module load Python/3.8.2-GCCcore-9.3.0

julia ./vae_one_class.jl $DATASET $CLASS
