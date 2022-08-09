#!/bin/bash
#SBATCH --time=24:00:00
#SBATCH --nodes=1 --ntasks-per-node=2 --cpus-per-task=2
#SBATCH --gres=gpu:1
#SBATCH --partition=gpu
#SBATCH --mem=40G
#SBATCH --job-name classifier
#SBATCH --output /home/skvarvit/logs/classifier/%J.log

DATASET=$1

module load Julia/1.5.3-linux-x86_64
module load Python/3.9.6-GCCcore-11.2.0

julia --project ./supervised_classifier.jl $DATASET
