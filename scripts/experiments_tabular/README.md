# Experiments - tabular data
To include a new model, first write a model specific run script. A good example is `knn.jl`. You have to provide 3 functions specific to to your model - `sample_params()`, that provides a sample of hyperparameters for the model constructor. Since some model parameters might be data-specific (e.g. depend on data dimensions), you have to overload the `GenerativeAD.edit_params(data, parameters)` function if needed. Finally, there is the `fit(data, parameters)` function that creates the model, fits it on training data and provides one or moe scoring functions.

Furthermore, create a model specific run script, such as `knn_run.sh`. Finally, you can evaluate the model on a single dataset with
```
./knn_run.sh dataset_name max_seed 
```
where `max_seed` is an integer that specifies the number of random crossvalidation retrainings.

To run a model in a parallel way across datasets, run
```
./run_parallel.sh run_script_name max_seed 
```
which parellelizes the run across a set number of CPU cores. Example:
```
./run_parallel.sh knn_run 5 
```
