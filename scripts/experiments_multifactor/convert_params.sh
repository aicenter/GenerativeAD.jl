#!/bin/bash
#SBATCH --partition cpufast
#SBATCH --nodes 1
#SBATCH --ntasks-per-node 1
#SBATCH --mem-per-cpu 40G
#SBATCH --time 4:00:00
#SBATCH --job-name convert-params
#SBATCH --output logs/convert_params-%J.log

MODELNAME=$1
DATASET=$2

module load Python/3.9.6-GCCcore-11.2.0
module load Julia/1.5.3-linux-x86_64

source ${HOME}/sgad-env/bin/activate
export PYTHON="${HOME}/sgad-env/bin/python"

julia --project ./convert_params.jl $MODELNAME $DATASET
