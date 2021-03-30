"""
Example script how to manually create Bayesian cache from already computed results. 
Intended to run interactively as user input is needed to rewrite parts of it.
Workflow:
    1. pick a model and dataset
    2. Define parameter ranges by `create_space` function.
    3. read corresponding folder from fisrt seed/anomaly_class
    4. filter incompatible runs that should not go into the cache
        (mainly if they parameters differ from what is currently defined in the run script) - define `reference_file_filter` 
        (optionally you can manualy convert old files into the new format) - define `result_postprocess!`
    5. randomly select some number of files - set `init_n`
    6. run the main forloop to create the cache
        uses built-in `register_run!` with `ignored_hyperparams` as input to indicate which parameters are not optimizable
    7. postprocess cache by filter based on extracted values - define `cache_postprocess!` 
    8. check if the conversions to_skopt and from_skopt work
    9. check if the bayesian optimizer can fit the values provided
    10. after this initial test, run it for all datasets

Bayesian cache structure
- dictionary indexed by `ophash` (hash of the named tuple containing only optimizable parameters)
- values - named tuples with following fields: parameters, seed, anomaly_class, runs

0x3212:
    parameters: 		named tuple containing optimizable parameters
    seed:				array of seeds on which models have been fitted
    anomaly_class:		array of anomaly_classes on which models have been fitted (store even for tabular datasets for better code readability)
    runs: 				array of floating point numbers containing computed metric for each run

example with `ocsvm`
0x1234: 
    (
        parameters = (gamma = 1.94199, nu = 0.34065, kernel = "rbf"), 
        seed = [1, 2, 3, 4, 5], 
        anomaly_class = [-1, -1, -1, -1, -1], 
        runs = [-0.92567, -0.92832, -0.94627, -0.93186, -0.95032]
    )
"""

using PyCall
using DrWatson
using Random
using BSON
using GenerativeAD
using ValueHistories
using LinearAlgebra

# If run outside our main repository - RECOMMENDED
DrWatson.projectdir() = "/home/skvarvit/generativead/GenerativeAD.jl"

### choose model, metric and dataset type
metric = :val_auc
modelname = "ocsvm"
prefix = "experiments/tabular"


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
    
### how many samples of fit results to take into account (50 by default)
### some models require more, as the procedure of cache creation may invalidate some
### for example RealNVP hyperparameters in Bayesian form do not allow for regularization 0.0
init_n = 50

"""
    reference_file_filter(files)

This filter is applied to result's folders.
In this example we throw out all model files (`!startswith(x, "model")`) and 
samples with old implementation `startswith("act")`.
Others are also possible such as filter only reconstruction sampled score with vae models
"""
function reference_file_filter(files)
    filter(startswith("act"), filter(x -> !startswith(x, "model"), files))
end
# default filters only models
reference_file_filter(files) = filter(x -> !startswith(x, "model"), files)

"""
    all_file_filter(files)

This filter is applied to an array of all files from a given dataset.
In this example we throw out all result files, that were trained on data seed>2.
"""
function all_file_filter(files)
    filter(x -> occursin("seed=1", x), files)
end
all_file_filter(files) = files

"""
    result_postprocess!(r)

This function is applied to every deserialized result file.
In this example we add `nu` parameter for runs that used the default `0.5`
but others such as filter only reconstruction sampled score with vae is used here.
"""
function result_postprocess!(r)
    if !(:nu in keys(r[:parameters]))
        p = merge(r[:parameters], (;nu=0.5))
        r[:parameters] = p
    end	
    r
end
result_postprocess!(r) = r 			# default - identity

### Before storing the parameter entry these fields are filtered.
### By default we don't want to optimize `init_seed` or `score_func`,
### though in the latter case this is only due to some code baggage.
ignored_hyperparams = Set([:init_seed])


"""
    cache_postprocess!(cache)

This function is applied to the resulting cache.
In this example we filter out samples with only one computed seed and samples
that have regularization turned off as it is not possible to model continuous parameter from [1f-6,1f-2] + 0.0f0
"""
function cache_postprocess!(cache)
    # filter out those that have only one seed
    filter!(c -> length(c[2][:runs]) > 1, cache)

    # filter out runs wih no regularization (throws domain errors from python)
    filter!(c -> c[2][:parameters][:wreg] > 0.0, cache)
end
cache_postprocess!(cache) = cache 	# default - identity


################                                                       ################
#######################################################################################

"""
downtherabithole(target)
Fetch file names from the first leaf node folder 
in the list of depthfirst recursively walked tree.
"""
function downtherabithole(target)
    entries = readdir(target, join=true)
    if all(isfile.(entries))
        return entries
    else
        return downtherabithole(entries[1])
    end
end


#######################################################################################
################            		MAIN LOOP 				           ################
#######################################################################################
# dataset = datasets[1]             # dry dry run
# for dataset in [datasets[1]]      # dry run
for dataset in datasets           # hot run

    # sample based on the first seed/anomaly_class using downtherabithole
    dataset_folder = joinpath(folder, dataset)
    reference_files = downtherabithole(dataset_folder)

    # filter files that are compatible
    reference_files_filtered = reference_file_filter(basename.(reference_files))

    # sample `init_n` files based on fixed seed
    n = length(reference_files_filtered)
    Random.seed!(7)
    files_to_load = Set(reference_files_filtered[randperm(n)][1:init_n])
    Random.seed!()
    
    # get all files and match only those that are in reference_files
    all_files = GenerativeAD.Evaluation.collect_files_th(dataset_folder);

    # additional file filter (for example when we don't want results from seed>2 for some image datasets)
    all_files_filtered = all_file_filter(all_files)
    
    # load files from each seed and call register_run! as will be done during actual training
    cache = Dict{UInt64, Any}()
    for f in all_files_filtered 
        if basename(f) in files_to_load
            r = BSON.load(f)

            # some manual repair may be needed
            r = result_postprocess!(r)
            try
                GenerativeAD.register_run!(
                    cache, r;
                    metric=:val_auc,
                    flip_sign=true,
                    ignore=ignored_hyperparams) # ignore hyperparameters
            catch e
                @warn("Register run on $(f) has failed due to $(e)")
            end
        end
    end
    @info "Cache creation completed: $(length(cache)) entries"
    
    cache_postprocess!(cache)
    @info "Cache postprocess completed: $(length(cache)) entries"

    ### The following code should not fail and create a valid set of hyperparameters.
    space = create_space()
    x0 = [v[:parameters] for v in values(cache)];
    y0 = [GenerativeAD.objective_value(v[:runs]) for v in values(cache)] 

    @info "Testing conversion from named tuples to skopt."
    x0s = [GenerativeAD.to_skopt(space, x) for x in x0]
    @info "Testing conversion back from skopt to named tuples."
    x0t = [GenerativeAD.from_skopt(space, x) for x in x0s]

    @info "Testing BayesianHyperOpt fit using the tell method."
    opt = GenerativeAD.BayesianHyperOpt(collect(space))
    GenerativeAD.tell(opt, x0s, y0)

    x1s = GenerativeAD.ask(opt)
    @info "Tested sampling of new parameters using ask: $x1s"

    x1 = GenerativeAD.from_skopt(space, x1s)
    @info "Tested conversion of sampled parameters from skopt to named tuples: $x1"

    # if all is well save the cache
    # by default this saves it back to the root directory
    DrWatson.projectdir() = pwd() 
    folder = datadir("$(prefix)/$(modelname)/$(dataset)")
    GenerativeAD.save_bayes_cache(folder, cache)
end