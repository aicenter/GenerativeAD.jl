using OrderedCollections

### this serves as a placeholder until bayesian optimization is merged into master
BAYES_CACHE = "bayes_cache.bson"
function load_bayes_cache(folder)
    file = joinpath(folder, BAYES_CACHE)
	isfile(file) ? BSON.load(file) : OrderedDict{UInt64, Any}()
end
###

"""
     combine_bayes(df, df_bayes; outer=true)

Enhances DataFrame `df_bayes` with results, that were used that were used in the 
construction of bayes caches (initial random samples). If `outer=true` then results 
from models that have not undergone bayesian optimization procedure are copied from 
corresponding random samplings runs in `df` (outer join).
"""
function combine_bayes(df, df_bayes; outer=true)
    models = outer ? unique(df.modelname) : unique(df_bayes.modelname)
    datasets = unique(df.dataset)
    dfs = map(Base.Iterators.product(datasets, models)) do (dataset, modelname)
        folder = datadir("experiments_bayes/tabular/$(modelname)/$(dataset)")
        cache = load_bayes_cache(folder)
        if length(cache) > 0
            phashes = unique(reduce(vcat, [c[:phashes] for c in values(cache)][1:50]))
            dff = copy(filter(x -> (x.modelname == modelname) && (x.dataset == dataset) && (x.phash in phashes), df))
            if nrow(dff) > 0
                @info "$modelname - $dataset - fetched $(nrow(dff)) rows"
            else
                @warn "$modelname - $dataset - could not fetch samples based on stored phashes"
            end
            dff
        else
            dff = copy(filter(x -> (x.modelname == modelname) && (x.dataset == dataset), df))
            @info "$modelname - $dataset - no bayes samples - fetching all $(nrow(dff)) rows"
            dff
        end
    end
    vcat(df_bayes, reduce(vcat, dfs))
end