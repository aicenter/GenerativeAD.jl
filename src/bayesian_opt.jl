using PyCall
using StatsBase

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
and y0 (objective).
"""
function tell(bho::BayesianHyperOpt, x0::Vector{Vector{Any}}, y0)
	bho.tell(x0, y0)
end

function ask(bho::BayesianHyperOpt)
	bho.ask()
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


# heavily in progress

# """
# 	function bayes_params(space, folder, fallback)
# Tries to load a cache of results to seed the optimizer with. The file should be in 

# """
# function bayes_params(space, folder, fallback)
# 	# load results
# 	results = load_bayes_cache
# 	if isfile("")

# 	else
# 		return fallback()
# 	end
# end

# function load_bayes_cache(modelname, dataset, prefix)

# 	x0, y0
# end

# """

# """
# function register_run()

# end