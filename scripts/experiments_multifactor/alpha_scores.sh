#!/bin/bash
#SBATCH --time=24:00:00
#SBATCH --nodes=1 --ntasks-per-node=2 --cpus-per-task=2
#SBATCH --partition=cpu
#SBATCH --mem=20G

MODELNAME=$1
DATASET=$2
LATENT_SCORE=$3
AC=$4
METHOD=$5
BASE_BETA=$6
AF1=$7
AF2=$8
AF3=$9

module load Julia/1.5.3-linux-x86_64
module load Python/3.9.6-GCCcore-11.2.0

source ${HOME}/sgad-env/bin/activate
export PYTHON="${HOME}/sgad-env/bin/python"
julia --project -e 'using Pkg; Pkg.build("PyCall"); @info("SETUP DONE")'

julia ./alpha_scores.jl ${MODELNAME} ${DATASET} ${LATENT_SCORE} $AC $METHOD ${BASE_BETA} --mf_normal --anomaly_factors $AF1 $AF2 $AF3
