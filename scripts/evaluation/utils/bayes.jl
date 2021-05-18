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
    df[:index] = collect(1:nrow(df))
    subsets = map(Base.Iterators.product(datasets, models)) do (dataset, modelname)
        folder = datadir("experiments_bayes/tabular/$(modelname)/$(dataset)")
        cache = load_bayes_cache(folder)
        if length(cache) > 0
            n = length(cache)
            phashes = unique(reduce(vcat, [c[:phashes] for c in values(cache)][1:min(50, n)]))
            indx = filter(x -> (x.modelname == modelname) && (x.dataset == dataset) && (x.phash in phashes), df)[:index]
            if length(indx) > 0
                @info "$modelname - $dataset - fetched $(length(indx)) rows with $(length(phashes)) hashes"
            else
                @warn "$modelname - $dataset - could not fetch samples based on stored phashes"
            end
            indx
        else
            indx = filter(x -> (x.modelname == modelname) && (x.dataset == dataset), df)[:index]
            @warn "$modelname - $dataset - no bayes samples - fetching all $(length(indx)) rows"
            indx
        end
    end
    subset = reduce(vcat, subsets[:])
    select!(df, Not(:index)) # drop index
    vcat(df_bayes, df[subset, :])
end