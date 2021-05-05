# GenerativeAD.jl
Generative models for anomaly detection. This Julia package contains code for the paper "Comparison of Anomaly Detectors: Context Matters" [arXiv preprint](https://arxiv.org/abs/2012.06260).

## Installation
1. Clone this repo somewhere.
2. Run Julia in the cloned dir.
```bash
cd path/to/repo/GenerativeAD.jl
julia --project
```
3. Install all the packages using `]instantiate` and compile the package.
```julia
(@julia) pkg> instantiate
(@julia) using GenerativeAD
```

Some of the bash scripts are calling `julia` without `--project` flag and uses `@quickactivate` macro to activate the environment, however this fails, unless `DrWatson` is installed in the base julia environment. In order to avoid these problems install `DrWatson` in your base environment.
```bash
cd ~
julia -e 'using Pkg; Pkg.add("DrWatson");'
```

### Python instalation
Some models (PIDforest, scikit-learn, PyOD) are available only through PyCall with appropriate environment active. With upcoming bayesian optimisation from `scikit-optimize` every model will require an active environment, which can be setup in following way using python's `venv` module. (Most of the scripts have hardcoded path to this environment, though this can be easily changed).
```bash
cd ~
python -m venv sklearn-env

source ${HOME}/sklearn-env/bin/activate
export PYTHON="${HOME}/sklearn-env/bin/python"
```
Then install requirements inside this repository
```bash
cd path/to/repo/GenerativeAD.jl

pip install -r requirements.txt
pip install git+https://github.com/janfrancu/pidforest.git # not registerd anywhere

julia --project -e 'using Pkg; Pkg.build("PyCall");' # rebuilds PyCall.jl to point to the current environment
```

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
