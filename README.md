# GenerativeAD.jl
Generative models for anomaly detection. This Julia package contains code for the paper "Comparison of Anomaly Detectors: Context Matters" [arXiv preprint](https://arxiv.org/abs/2012.06260).

## Installation

Install prerequisites:
```julia
(@julia) pkg> add https://github.com/vitskvara/UCI.jl.git
(@julia) pkg> add https://github.com/pevnak/SumProductTransform.jl.git
(@julia) pkg> add https://github.com/janfrancu/ContinuousFlows.jl.git
```

Then install the package itself:

1. Clone this repo somewhere.
2. Run Julia in the cloned dir.
```bash
cd path/to/repo/GenerativeAD.jl
julia --project
```
3. Install all the packages and compile the package.
```julia
(@julia) pkg> add DrWatson
(@julia) pkg> instantiate
(@julia) using GenerativeAD
```

## Experimental setup:

To implement a new model, you need to define methods for model construction, fitting and prediction. For details, see e.g. the readme in `scripts/experiments_tabular`, where the experimental setup for running experiment repetitions is explained.

## Running experiments on the RCI cluster

0. First, load Julia and Python modules.
```bash
ml Julia
ml Python
```
1. Install the package somewhere on the RCI cluster.
2. Then the experiments can be run via `slurm`. This will run 20 experiments with the basic VAE model, each with 5 crossvalidation repetitions on all datasets in the text file with 10 parallel processes for each dataset. All data will be saved in `GenerativeAD.jl/data/experiments/tabular`
```bash
cd GenerativeAD.jl/scripts/experiments_tabular
./run_parallel.sh vae 20 5 10 datasets_tabular.txt
```

## Data:

Only UCI datasets are available upon installation via the `UCI` package. Remaining tabular and image datasets are downloaded upon first request (e.g. via the `GenerativeAD.Datasets.load_data(dataset)` function). First download requires user input to accept download terms for individual datasets. If you want to avoid this, do
```bash
export DATADEPS_ALWAYS_ACCEPT=true
```
before running Julia.
