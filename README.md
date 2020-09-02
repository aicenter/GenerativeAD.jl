# GenerativeAD.jl
Generative models for anomaly detection.

## Installation

Install prerequisites:
```julia
(@julia) pkg> add https://github.com/vitskvara/UCI.jl.git
```

Then install the package itself:
```julia
(@julia) pkg> add https://github.com/aicenter/GenerativeAD.jl.git
```

## Experimental setup:

1) Each model has methods `constructor(args**, kwargs**)`, `train!(model, data)`, `fit(model, data)` and `sample_params()`.
2) On a single dataset, run each model on a limited budget, e.g. 100 random hyperparameter settings or limited time.
3) Do train/validation/test split with a fixed seed.
4) Crossvalidation only for the best model? Or not at all? 10 folds would be very expensive, especially on image data.
5) Save almost everything - fit, train time, input parameters including the seed used for dataset splitting, scores on train/validation/test samples, labels. Model state as well? May be extremely space consuming but could be useful for computing additional scores.
6) Evaluation metrics are computed post-hoc because they are cheap and their list will probably change a lot. Definitelly use AUC, AUPRC, F1, TPR@{1,2,5,10}, pAUC@{1,2,5,10}, maybe bootstrap TPR@?
7) Try to run models all the time, while different models are being developed.
