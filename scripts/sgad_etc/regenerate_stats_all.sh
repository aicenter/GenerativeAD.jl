#!/bin/bash
for MODEL in vae DeepSVDD fmgan fAnoGAN
do
	julia --project regenerate_stats.jl /home/skvarvit/generativead/GenerativeAD.jl $MODEL CIFAR10
	julia --project regenerate_stats.jl /home/skvarvit/generativead/GenerativeAD.jl $MODEL SVHN2
	julia --project regenerate_stats.jl /home/skvarvit/generativead-sgad/GenerativeAD.jl $MODEL wildlife_MNIST
	julia --project regenerate_stats.jl /home/skvarvit/generativead/GenerativeAD.jl $MODEL transistor
	julia --project regenerate_stats.jl /home/skvarvit/generativead-sgad/GenerativeAD.jl $MODEL pill
	julia --project regenerate_stats.jl /home/skvarvit/generativead-sgad/GenerativeAD.jl $MODEL metal_nut
	julia --project regenerate_stats.jl /home/skvarvit/generativead-sgad/GenerativeAD.jl $MODEL capsule
	julia --project regenerate_stats.jl /home/skvarvit/generativead-sgad/GenerativeAD.jl $MODEL bottle
done

for MODEL in cgn sgvae
do
	julia --project regenerate_stats.jl /home/skvarvit/generativead-sgad/GenerativeAD.jl $MODEL CIFAR10
	julia --project regenerate_stats.jl /home/skvarvit/generativead-sgad/GenerativeAD.jl $MODEL SVHN2
	julia --project regenerate_stats.jl /home/skvarvit/generativead-sgad/GenerativeAD.jl $MODEL wildlife_MNIST
	julia --project regenerate_stats.jl /home/skvarvit/generativead-sgad/GenerativeAD.jl $MODEL transistor
	julia --project regenerate_stats.jl /home/skvarvit/generativead-sgad/GenerativeAD.jl $MODEL pill
	julia --project regenerate_stats.jl /home/skvarvit/generativead-sgad/GenerativeAD.jl $MODEL metal_nut
	julia --project regenerate_stats.jl /home/skvarvit/generativead-sgad/GenerativeAD.jl $MODEL capsule
	julia --project regenerate_stats.jl /home/skvarvit/generativead-sgad/GenerativeAD.jl $MODEL bottle
done

