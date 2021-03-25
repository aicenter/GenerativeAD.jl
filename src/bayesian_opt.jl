using PyCall
using StatsBase
using Statistics

using .Evaluation: compute_stats, _get_anomaly_class

const BAYES_CACHE = "bayes_cache.bson"
"""
	BayesianHyperOpt

Wrapper around `skotp.optimizer.Optimizer`. 
Uses default params from `skopt.gp_minimize`.
Constructed using a list/array of `skotp.space`'s classes.
"""
struct BayesianHyperOpt
	opt
	function BayesianHyperOpt(dimensions) 
		py"""
		import numpy as np
		from skopt import Optimizer
		from skopt.utils import cook_estimator, normalize_dimensions
		from sklearn.utils import check_random_state

		def construct_optimizer(dimensions):
			rng = check_random_state(7) 			# fix seed to fit always the same model
			space = normalize_dimensions(dimensions)

			base_estimator = cook_estimator(
							"GP", space=space, 
							random_state=rng.randint(0, np.iinfo(np.int32).max),
							noise="gaussian")

			# default from gp_minimize
			acq_optimizer_kwargs = {"n_points": 10000, "n_restarts_optimizer": 5, "n_jobs": 1}
			acq_func_kwargs = {"xi": 0.01, "kappa": 1.96}

			optimizer = Optimizer(
				space,
				base_estimator,
				n_initial_points=0, 				# should be zero in order to skip the random generation
				initial_point_generator="random", 	# it fallbacks into random even if it is None, so here you go
				n_jobs=-1,
				acq_func="gp_hedge", 				# default from gp_minimize
				acq_optimizer="auto", 				# default from gp_minimize
				random_state=rng,
				model_queue_size=None,				# default from gp_minimize
				acq_optimizer_kwargs=acq_optimizer_kwargs,
				acq_func_kwargs=acq_func_kwargs)

			return optimizer
		"""
		new(py"construct_optimizer"(dimensions))
	end
end

"""
	function tell(bho::BayesianHyperOpt, x0, y0)

Fits the optimizer based on multiple observations of x0 (hyperparameter) 
and y0 (objective). Be careful about the input types here as conversions
to Python are tricky. By default `x0` is passed as list of lists, however
vector of values `y0` is converted to `numpy.ndarray`, which cannot 
be processed by `skopt` (!! What a nice ecosystem !!).
"""
function tell(bho::BayesianHyperOpt, x0::Vector{Vector{Any}}, y0)
	bho.opt.tell(x0, Tuple(y0))
end

function ask(bho::BayesianHyperOpt)
	bho.opt.ask()
end


"""
	skopt_parse(dimension, entry::T)
Converts and entry of a hyperparameter from skotp dimension to named tuple.
Dimension's name have to following this convention:

```py
	Integer(1, 3, name="log2_{name}") # will convert to 2^p
	Integer(1, 3, name="log10_{name}") # will convert to 10^p
	Categorical(categories = [:dense, :all, :none], name="sym_{name}") # will convert to and from Symbol
	Categorical(categories = [identity], name="fun_{name}") # will convert to and from a function name
```
Other cases of `T` such as Bool, Real, Int are handled without any specific treatment.
"""
skopt_parse(dimension, entry::Bool) = (;[Symbol(dimension.name) => entry]...)
skopt_parse(dimension, entry::T) where {T <: AbstractFloat} = (;[Symbol(dimension.name) => entry]...)

function skopt_parse(dimension, entry::T) where {T <: Integer}
	if startswith(dimension.name, "log2_")
		return (;[Symbol(dimension.name[6:end]) => 2^entry]...)
	elseif startswith(dimension.name, "log10_")
		return (;[Symbol(dimension.name[7:end]) => 10^entry]...)
	else
		return (;[Symbol(dimension.name) => entry]...)
	end
end

function skopt_parse(dimension, entry::T) where {T <: AbstractString}
	if startswith(dimension.name, "sym_") 
		return (;[Symbol(dimension.name[5:end]) => Symbol(entry)]...)
	elseif startswith(dimension.name, "fun_")
		return (;[Symbol(dimension.name[5:end]) => eval(:($(Symbol(entry))))]...)
	else
		return (;[Symbol(dimension.name) => entry]...)
	end
end

"""
	from_skopt(space, p)
Converts an array of hyperparameters `p` from skotp to a named tuple 
based on provided metadata in skotp's Space class instance `space`.
The order of entries in `p` coincides with the order of dimensions in `space`, 
if that is not the case something broke down.
"""
function from_skopt(space, p)
	merge([skopt_parse(dim, entry) for (dim, entry) in zip(space, p)]...)
end


tuple_parse(dimension, entry::Bool) = entry
tuple_parse(dimension, entry::T) where {T <: AbstractString} = entry
tuple_parse(dimension, entry::T) where {T <: AbstractFloat} = entry
tuple_parse(dimension, entry::Symbol) = string(entry)
tuple_parse(dimension, entry::Function) = string(entry) 	# SPTN has identity function in hyperparameters

function tuple_parse(dimension, entry::T) where {T <: Integer} 
	if startswith(dimension.name, "log2")
		return Int(log2(entry))
	elseif startswith(dimension.name, "log10")
		return Int(log10(entry))
	else
		return entry
	end
end

"""
	to_skopt(space, p)
Converts named tuple of hyperparameters `p` from Julia to a list of hyperparameters in skopt 
based on provided metadata in skotp's Space class instance `space`. 
The order of entries in `p` does not have to coincide with the order of dimensions in `space`, 
but it will be imposed by keys of the `space` tuple.
"""
function to_skopt(space, p)
	[tuple_parse(dim, p[name]) for (name, dim) in pairs(space)]
end


"""
	objective_value(r)

Validates entry from bayesian cache and returns aggregated score, which is optimized by `BayesianHyperOpt`.
For empty entries (no model trained) we return `fail_value`.
"""
function objective_value(runs; agg=mean, fail_value=0.0f0)
	if length(runs) > 0 # should we be more strict?
		return agg(runs)
	else
		return fail_value
	end
end

"""
	function bayes_params(space, folder, parameter_range; add_model_seed=true)
Tries to load a cache of results to seed the optimizer with. The file should be named by `BAYES_CACHE`
global constant and located in `folder`. This method fallbacks to using `sample_params` on
provided `parameter_range`, if the cache has not been created. 
"""
function bayes_params(space, folder, parameter_range; add_model_seed=true)
	cache = load_bayes_cache(folder)
	if cache !== nothing		
		x0 = [v[:parameters] for v in values(cache)]
		y0 = [objective_value(v[:runs]) for v in values(cache)] 
		@info("Loaded cached results from $(folder).")
		x0 = [to_skopt(space, x) for x in x0]
		opt = BayesianHyperOpt(collect(space))
		tell(opt, x0, y0)
		@info("Fitted GP with $(length(x0)) samples.")
		p = from_skopt(space, ask(opt))
		
		# registers only parameters with empty arrays and store cache
		register_run!(cache, (;parameters=p); fit_results=false)
		save_bayes_cache(folder, cache)

		@info("Fetched new hyperparameters: $(p).")
		return add_model_seed ? merge((;init_seed=rand(1:Int(1e8))), p) : p
	else
		@warn("Could not find bayesian cache file at $(folder), sampling.")
		return sample_params(parameter_range, add_model_seed=add_model_seed)
	end
end

"""
	load_bayes_cache(folder)

Tries to load serialized results of Bayesian optimization from `folder`.
"""
function load_bayes_cache(folder)
    file = joinpath(folder, BAYES_CACHE)
	isfile(file) ? BSON.load(file) : Dict{UInt64, Any}()
end

"""
	save_bayes_cache(folder)

Saves serialized results of Bayesian optimization to `folder`.
"""
function save_bayes_cache(folder, cache)
    file = joinpath(folder, BAYES_CACHE)
	wsave(file, cache)
end

"""
	update_bayes_cache(folder, results,  )

Loads bayesian cache from folder and updates it with results
TODO Updating with multiple results is not working yet - could be done
by giving `register_run!` dataframe with aggregated columns using aggregation in `agg`.
"""
function update_bayes_cache(folder, results, agg=:max; kwargs...)
	_cache = load_bayes_cache(folder) 
	cache = _cache !== nothing ? _cache : Dict{UInt64, Any}()
	for r in results
		register_run!(cache, r; kwargs...)
	end
	save_bayes_cache(folder, cache)
end

"""
	register_run!(cache, r::Dict{Symbol,Any}; metric=:val_auc, ignore=Set([:init_seed]))

Updates `cache` with new result from `r`. All basic metrics from `.Evaluation` module are
supported and their column name can be passed in the `metric` argument.
Set of hyperparameters that are not optimized is given by the `ignore` argument, 
which by default filters `:init_seed`. 
"""
function register_run!(cache, r::Dict{Symbol,Any}; fit_results=true, metric=:val_auc, flip_sign=true, ignore=Set([:init_seed]))
	metric_value = try
		fit_results ? compute_stats(r, top_metrics=false)[metric] : []
	catch e
		@warn("Cache has not been updated. Stats computation failed due to $e")
		return
	end
	
	# filter only those parameters that are being optimized
	parameters = (;filter(p -> !(p[1] in ignore), pairs(r[:parameters]))...) 
	ophash = hash(parameters)   # corresponds only to optimizable parameters
								# there may be colisions (e.g. parameter being run with two seeds)

	if ophash in keys(cache) && fit_results				# add to existing entry
		entry = cache[ophash]
		seed = vcat(entry[:seed], r[:seed])
		anomaly_class = vcat(entry[:anomaly_class], _get_anomaly_class(r))
		runs = vcat(entry[:runs], flip_sign ? -metric_value : metric_value)

		cache[ophash] = merge(entry, (;seed=seed, anomaly_class=anomaly_class, runs=runs))
	elseif !(ophash in keys(cache)) && fit_results		# or create new entry if there are results (mainly used during manual cache creation)
		entry = (;
			parameters = parameters,
			seed = [r[:seed]],
			anomaly_class = [_get_anomaly_class(r)],
			runs = flip_sign ? -metric_value : metric_value)
		cache[ophash] = entry
	else							# otherwise add an empty entry (this is to indicate that such parameters are about to be trained)
		entry = (;
			parameters = parameters,
			seed = Int[],
			anomaly_class = Int[],
			runs = Float32[])
		
		cache[ophash] = entry
	end
end