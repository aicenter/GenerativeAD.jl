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
```julia
(@julia) pkg> add https://github.com/aicenter/GenerativeAD.jl.git
```
and instantiate from the package directory to install all the 

## Experimental setup:

To implement a new model, you need to define methods for model construction, fitting and prediction. For details, see e.g. the readme in `scripts/experiments_tabular`, where the experimental setup for running experiment repetitions is explained.

### Data:

Only UCI datasets are available upon installation via the `UCI` package. Remaining tabular and image datasets are downloaded upon first request (e.g. via the `GenerativeAD.Datasets.load_data(dataset)` function). First download requires user input to accept download terms for individual datasets. If you want to avoid this, do
```bash
export DATADEPS_ALWAYS_ACCEPT=true
```
before running Julia.