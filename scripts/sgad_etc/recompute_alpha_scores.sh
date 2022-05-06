#!/bin/bash
#SBATCH --time=4:00:00
#SBATCH --nodes=1 --ntasks-per-node=2 --cpus-per-task=2
#SBATCH --partition=cpufast
#SBATCH --mem=30G

DATASET=$1
DATATYPE=$2
LATENT_SCORE=$3
METHOD=$4
FORCE=$5

module load Julia/1.5.3-linux-x86_64
module load Python/3.9.6-GCCcore-11.2.0

source ${HOME}/sgad-env/bin/activate
export PYTHON="${HOME}/sgad-env/bin/python"
julia --project -e 'using Pkg; Pkg.build("PyCall"); @info("SETUP DONE")'

julia ./recompute_alpha_scores.jl sgvae ${DATASET} ${DATATYPE} ${LATENT_SCORE} ${METHOD} $FORCE
