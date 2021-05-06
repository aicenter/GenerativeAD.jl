"""
Example script how to manually create Bayesian cache from already computed results. 
Intended to run interactively as user input is needed to rewrite parts of it.
Workflow:
    0. Install `scikit-optimize` 
        (from now on the environment that this is installed into has to be the one to activate before each run script)
    1. pick a model and dataset
    2. Define parameter ranges by `create_space` function.
    3. read recursively all results from corresponding dataset folder
    4. run the main forloop to create the cache
        uses built-in `register_run!` with `ignored_hyperparams` as input to indicate which parameters are not optimizable
        patching of individual results is possible using `result_postprocess!`
    5. postprocess cache by filter based on extracted values - define `cache_postprocess!` 
    6. sample `init_n` entries from cache with fixed seed
    7. check if the conversions to_skopt and from_skopt work
    8. check if the bayesian optimizer can fit the values provided
    9. after this initial test, run it for all datasets

Bayesian cache structure
- !ordered! dictionary indexed by `ophash` (hash of the named tuple containing only optimizable parameters)
- values - named tuples with following fields: parameters, runs, phashes

ophash:
    parameters: 		named tuple containing optimizable parameters
    runs: 				dictionary indexed by (seed, anomaly_class) containing computed metric for each run and scores
    phashes:            array of hashes of full parameters (this is used to find the original files during evaluation)

example with `ocsvm`
0x1234: 
    (
        parameters = (gamma = 1.94199, nu = 0.34065, kernel = "rbf"), 
        runs = Dict(
            (1,-1) => [-0.92567], 
            (2,-1) => [-0.92832], 
            (3,-1) => [-0.94627], 
            (4,-1) => [-0.93186], 
            (5,-1) => [-0.95032]),
        phashes = [0x123456, 0x123465, 0x123478, 0x123487, 0x123489]
    )
"""

using PyCall
using DrWatson
using Random
using BSON
using GenerativeAD
using ValueHistories
using LinearAlgebra
using OrderedCollections
using Suppressor

# If run outside our main repository - RECOMMENDED
DrWatson.projectdir() = "/home/skvarvit/generativead/GenerativeAD.jl"

### choose model, metric and dataset type
metric = :val_auc
modelname = "fill in modelname"
prefix = "experiments/tabular"
target_prefix = "experiments_bayes/tabular"

### how many samples of fit results to take into account (50 by default)
init_n = 50

folder = datadir("$(prefix)/$(modelname)")
datasets = readdir(folder)

#######################################################################################
################ THIS PART IS TO BE PROVIDED FOR EACH MODEL SEPARATELY ################
"""
    create_space()

Creates a named tuple of python `skotp.space` classes which corresponds 
to the hyperparameter ranges in each script.
Some naming conventions apply, see `?GenerativeAD.skopt_parse`.
Copy this from/to the corresponds run script.
"""
function create_space()
    pyReal = pyimport("skopt.space")["Real"]
    pyInt = pyimport("skopt.space")["Integer"]
    pyCat = pyimport("skopt.space")["Categorical"]
    
    (;
    gamma 		= pyReal(1e-4, 1e2, prior="log-uniform",		name="gamma"),
    kernel 		= pyCat(categories=["poly", "rbf", "sigmoid"],	name="kernel"),
    nu 			= pyReal(0.01, 0.99, 							name="nu")
    )
end
    

"""
    file_filter!(files)

This filter is applied to an array of all files from a given dataset.
In this first example we throw out all result files, that were trained on data seed>2 and
we also usually want to filter out model files. 
(image datasets still have some samples computed on seed>2)
"""
file_filter(files) = filter(x -> occursin("seed=1", x) && !occursin("model", x), files)
file_filter(files) = filter(x -> !occursin("model", x), files) # default - behavior

### Before storing the parameter entry these fields are filtered.
### By default we don't want to optimize `init_seed` or parameters related to anomaly score.
### this has to be copied into `register_run!` in the corresponding run script.
ignored_hyperparams = Set([:init_seed, :percentile])


"""
    cache_postprocess!(cache)

This function is applied to the resulting cache.
In this example we filter out samples with only one computed seed and samples
that have regularization turned off as it is not possible to model continuous parameter from [1f-6,1f-2] + 0.0f0
"""
function cache_postprocess!(cache)
    # filter out those that have only one seed
    filter!(c -> length(c[2][:runs]) > 1, cache)

    # filter out runs with no regularization (throws domain errors from python)
    filter!(c -> c[2][:parameters][:wreg] > 0.0, cache)

    # filter out runs on the old implementation
    filter!(c -> :act_loc in keys(c[2][:parameters]), cache)
end
cache_postprocess!(cache) = filter!(c -> length(c[2][:runs]) > 1, cache) # default filter


################                                                       ################
#######################################################################################

#######################################################################################
################            		MAIN LOOP 				           ################
#######################################################################################
# dataset = datasets[1]             # dry dry run
# for dataset in [datasets[1]]      # dry run
for dataset in datasets           # hot run
  
    @info "Processing result of $modelname on $dataset."
    dataset_folder = joinpath(folder, dataset)
    
    # list recursively all files
    files = GenerativeAD.Evaluation.collect_files_th(dataset_folder);
    @info "Collected all $(length(files)) files from $(dataset_folder) folder."

    # file filter (for example when we don't want results from seed>2 for image datasets)
    files_filtered = file_filter(files)
    @info "Applied filter: currently left $(length(files_filtered)) files."
    
    @info "Running cache builder."
    # load files from each seed and call register_run! in the same way as after training
    cache = OrderedDict{UInt64, Any}()
    for f in files_filtered 
        r = BSON.load(f)

        try
            @suppress begin
                GenerativeAD.register_run!(
                    cache, r;
                    metric=:val_auc,
                    flip_sign=true,
                    ignore=ignored_hyperparams) # ignore hyperparameters
            end
        catch e
            @warn("Register run on $(f) has failed due to $(e)")
        end
    end
    @info "Cache creation completed: $(length(cache)) entries"
    if length(cache) > 140 
        @warn "There may be too many entries in the cache. Check `ignored_hyperparams`."
    end
    
    cache_postprocess!(cache)
    @info "Cache postprocess completed: $(length(cache)) entries"

    n = length(cache)
    Random.seed!(7);
    mask = Set(randperm(n)[1:min(init_n, n)])
    cache = OrderedDict(k => v for (i, (k,v)) in enumerate(cache) if i in mask)
    Random.seed!();
    @info "Sampling completed: $(length(cache)) entries"

    ### The following code should not fail and create a valid set of hyperparameters.
    space = create_space()
    x0 = [v[:parameters] for v in values(cache)];
    y0 = [GenerativeAD.objective_value(v[:runs]) for v in values(cache)] 

    @info "Testing conversion from named tuples to skopt."
    x0s = [(GenerativeAD.to_skopt(space, x)..., ) for x in x0]
    @info "Testing conversion back from skopt to named tuples."
    x0t = [GenerativeAD.from_skopt(space, x) for x in x0s]

    @info "Testing BayesianHyperOpt fit using the tell method."
    try
        opt = GenerativeAD.BayesianHyperOpt(collect(space))
        GenerativeAD.tell(opt, x0s, y0)
        
        x1s = GenerativeAD.ask(opt)
        @info "Tested sampling of new parameters using ask: $x1s"
        
        x1 = GenerativeAD.from_skopt(space, x1s)
        @info "Tested conversion of sampled parameters from skopt to named tuples: $x1"
    catch e
        # helpful for debugging problems with definition of space
        # all hyperparameter dimensions must be contained in their bounds
        @warn "Failed during dry run of optimization due to $e"
        @info "This may be due to some points out of bounds showing debug:"
        @info("", [(x, [py"$(x[i]) in $(p)" for (i, p) in enumerate(space)]) for x in x0s])
        break
    end

    # if all is well save the cache
    # by default this saves it back to the root directory
    DrWatson.projectdir() = pwd() 
    target_folder = datadir("$(target_prefix)/$(modelname)/$(dataset)")
    @info "Saving cache to $(target_folder)"
    GenerativeAD.save_bayes_cache(target_folder, cache)
    @info "________________________________________________________________________________"
end
