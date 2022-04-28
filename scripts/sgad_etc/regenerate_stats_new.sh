#!/bin/bash
#SBATCH --partition=cpufast
#SBATCH --time=4:00:00
#SBATCH --nodes=1 --ntasks-per-node=2 --cpus-per-task=2
#SBATCH --mem=20G

MODEL=$1
FORCE=$2

module load Julia/1.5.3-linux-x86_64
module load Python/3.9.6-GCCcore-11.2.0

source ${HOME}/sgad-env/bin/activate
export PYTHON="${HOME}/sgad-env/bin/python"
julia --project -e 'using Pkg; Pkg.build("PyCall"); @info("SETUP DONE")'

julia --project regenerate_stats.jl /home/skvarvit/generativead-sgad/GenerativeAD.jl $MODEL CIFAR10 $FORCE
julia --project regenerate_stats.jl /home/skvarvit/generativead-sgad/GenerativeAD.jl $MODEL SVHN2 $FORCE
julia --project regenerate_stats.jl /home/skvarvit/generativead-sgad/GenerativeAD.jl $MODEL wildlife_MNIST $FORCE
julia --project regenerate_stats.jl /home/skvarvit/generativead-sgad/GenerativeAD.jl $MODEL transistor $FORCE
julia --project regenerate_stats.jl /home/skvarvit/generativead-sgad/GenerativeAD.jl $MODEL pill $FORCE
julia --project regenerate_stats.jl /home/skvarvit/generativead-sgad/GenerativeAD.jl $MODEL metal_nut $FORCE
julia --project regenerate_stats.jl /home/skvarvit/generativead-sgad/GenerativeAD.jl $MODEL capsule $FORCE
julia --project regenerate_stats.jl /home/skvarvit/generativead-sgad/GenerativeAD.jl $MODEL bottle $FORCE
