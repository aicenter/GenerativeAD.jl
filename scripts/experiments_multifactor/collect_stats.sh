#!/bin/bash
#SBATCH --time=24:00:00
#SBATCH --nodes=1 --ntasks-per-node=2 --cpus-per-task=2
#SBATCH --partition=cpu
#SBATCH --mem=20G

AC=$1

module load Julia/1.5.3-linux-x86_64
module load Python/3.9.6-GCCcore-11.2.0

source ${HOME}/sgad-env/bin/activate
export PYTHON="${HOME}/sgad-env/bin/python"
julia --project -e 'using Pkg; Pkg.build("PyCall"); @info("SETUP DONE")'

julia --threads 16 --project ../evaluation/collect_stats.jl experiments_multifactor/alpha_evaluation_mf_normal/sgvae/wildlife_MNIST/ac=$AC experiments_multifactor/alpha_evaluation_mf_normal/images_leave-one-in_eval_ac=$AC.bson -f
