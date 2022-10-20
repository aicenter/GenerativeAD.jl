#!/bin/bash
#SBATCH --time=24:00:00
#SBATCH --nodes=1 --ntasks-per-node=2 --cpus-per-task=2
#SBATCH --partition=cpu
#SBATCH --mem=30G

MODEL=$1
DATASET=$2
LATENT_SCORE=$3
ANOMALY_CLASS=$4
BASE_BETA=$5
VAL_CLASSES=$6
FORCE=$7

module load Julia/1.5.3-linux-x86_64
module load Python/3.9.6-GCCcore-11.2.0

source ${HOME}/sgad-env/bin/activate
export PYTHON="${HOME}/sgad-env/bin/python"
julia --project -e 'using Pkg; Pkg.build("PyCall"); @info("SETUP DONE")'

julia ./alpha_scores.jl $MODEL ${DATASET} ${LATENT_SCORE} ${ANOMALY_CLASS} ${BASE_BETA} ${VAL_CLASSES} $FORCE
