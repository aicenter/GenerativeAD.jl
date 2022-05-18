#!/bin/bash
#SBATCH --time=4:00:00
#SBATCH --nodes=1 --ntasks-per-node=2 --cpus-per-task=2
#SBATCH --partition=cpufast
#SBATCH --mem=20G

MODELNAME=$1
DATASET=$2
TRAIN_CLASS=$3
AF1=$4
AF2=$5
AF3=$6

module load Julia/1.5.3-linux-x86_64
module load Python/3.9.6-GCCcore-11.2.0

source ${HOME}/sgad-env/bin/activate
export PYTHON="${HOME}/sgad-env/bin/python"
julia --project -e 'using Pkg; Pkg.build("PyCall"); @info("SETUP DONE")'

julia ./multifactor_experiment.jl ${MODELNAME} ${TRAIN_CLASS} ${DATASET} -f --anomaly_factors $AF1 $AF2 $AF3
