#!/bin/bash
#SBATCH --time=24:00:00
#SBATCH --nodes=1 --ntasks-per-node=2 --cpus-per-task=2
#SBATCH --partition=cpu
#SBATCH --mem=30G

DATASET=$1
LATENT_SCORE=$2
ANOMALY_CLASS=$3
BASE_BETA=$4
FORCE=$5

module load Julia/1.5.3-linux-x86_64
module load Python/3.9.6-GCCcore-11.2.0

source ${HOME}/sgad-env/bin/activate
export PYTHON="${HOME}/sgad-env/bin/python"
julia --project -e 'using Pkg; Pkg.build("PyCall"); @info("SETUP DONE")'

julia ./alpha_scores.jl sgvae ${DATASET} ${LATENT_SCORE} ${ANOMALY_CLASS} ${BASE_BETA} $FORCE
