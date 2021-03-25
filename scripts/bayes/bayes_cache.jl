"""
Example script how to manually create Bayesian cache with already computed results.
Workflow:
	1. pick a model and dataset
	2. read corresponding folder from fisrt seed/anomaly_class
	3. filter incompatible runs that should not go into the cache (mainly if they parameters differ from what is currently defined in the run script)
	 	(optionally you can manualy convert old files into the new format) 
	4. randomly select some number of files
	5. run the main forloop to create the cache
	6. check if the conversions to_skopt and from_skopt work
	7. check if the bayesian optimizer can fit the values provided
	8. after this initial test, run it for all datasets
"""



using DrWatson
using Random
using BSON
using GenerativeAD
using ValueHistories
using LinearAlgebra

DrWatson.projectdir() = "/home/skvarvit/generativead/GenerativeAD.jl"

### tabular example
metric = :val_auc
modelname = "RealNVP"
dataset = "abalone"
prefix = "experiments/tabular"

### image example TODO
# metric = :val_auc
# modelname = "fmgan"
# dataset = "CIFAR10"
# prefix = "experiments/images"

folder = datadir("$(prefix)/$(modelname)/$(dataset)")

# sample based on the first seed and load the same names from other seeds
# first seed is guaranteed to have the most number of experiments
seed_folders = readdir(folder, join=true)
seed_files = readdir(seed_folders[1])

# custom filter for each model
seed_files = filter!(startswith("act"), filter(x -> !startswith(x, "model"), seed_files))

n = length(seed_files)
Random.seed!(7)
files_to_load = seed_files[randperm(n)][1:75] # tune this number to get around 50 in the end
Random.seed!()


# load files from each seed and call register_run! as will be done during actual training
cache = Dict{UInt64, Any}()
for f in files_to_load
	for s in 1:5
		file = joinpath(folder, "seed=$s", f)
		if isfile(file)
			r = BSON.load(file)
			try
				GenerativeAD.register_run!(
					cache, r; 
					metric=:val_auc,
					flip_sign=true,
					ignore=Set([:init_seed])) # add score filtration for some models
			catch e
				@warn("Register run has failed due to $(e)")
			end
		end
	end
end
cache

# cache postprocess
# filter out those that have only one seed
cache = filter(c -> length(c[2][:runs]) >= 3, cache)

# filter out runs wih no regularization (throws domain errors from python)
cache = filter(c -> c[2][:parameters][:wreg] > 0.0, cache)

# check compatibility with space defined in run scripts' `create_space()`
# running 
space = create_space()

x0 = [v[:parameters] for v in values(cache)];
x0s = [GenerativeAD.to_skopt(space, x) for x in x0]
x0t = [GenerativeAD.from_skopt(space, x) for x in x0s]
y0 = [GenerativeAD. for v in values(cache)]

opt = GenerativeAD.BayesianHyperOpt(collect(space))
GenerativeAD.tell(opt, x0s, y0)
x1s = GenerativeAD.ask(opt)
GenerativeAD.from_skopt(space, x1s)


# if all is well save the cache
GenerativeAD.save_bayes_cache(folder, cache)