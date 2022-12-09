#!/bin/bash
#SBATCH --gres=gpu:1
#SBATCH --partition gpufast
#SBATCH --nodes 1
#SBATCH --ntasks-per-node 1
#SBATCH --mem-per-cpu 40G
#SBATCH --time 4:00:00
#SBATCH --job-name jupyter-gpu
#SBATCH --output logs/%J.log

CONFIG=$1

# get tunneling info
module load Python/3.9.6-GCCcore-11.2.0
module load Julia/1.5.3-linux-x86_64

source ${HOME}/sgad-env/bin/activate
export PYTHON="${HOME}/sgad-env/bin/python"

julia --project sgvaegan100.jl $CONFIG
