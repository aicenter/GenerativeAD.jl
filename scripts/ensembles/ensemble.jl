using ArgParse
using DrWatson
@quickactivate
using BSON
using Random
using FileIO
using DataFrames
using Base.Threads: @threads
using GenerativeAD

using GenerativeAD.Evaluation: _prefix_symbol, compute_stats

# pkgs which come from deserialized BSONs
# have to be present in the Main module
using ValueHistories
using LinearAlgebra

s = ArgParseSettings()
@add_arg_table! s begin
    "modelname"
		arg_type = String
		default = "sptn"
		help = "Name of the model from which to compute ensemble."
	"dataset"
		arg_type = String
		default = "iris"
		help = "Dataset name."
    "dataset_type"
		arg_type = String
		default = "tabular"
        help = "images | tabular"
    "max_seed"
        arg_type = Int
        default = 5
        help = "Number of seeds to go through with each dataset."
end


### testing
DrWatson.projectdir() = "/home/skvarvit/generativead/GenerativeAD.jl"
modelname = "MAF"
dataset = "iris"
seed = 1
dataset_type = "tabular"
###

function ensemble_experiment(modelname, dataset, dataset_type, seed; kwargs...)
    # this has to change for images
    # would be simpler if loaded from evaluation
    # filenames are preserved there so the lookup is fast
    failed = zeros(Bool, length(files))
    directory = datadir("experiments/$(dataset_type)/$(modelname)/$(dataset)/seed=$(seed)/")
    files = readdir(directory, join=true)
    filter!(x -> !startswith(basename(x), "model_"), files)

    rows = []
    for (i,f) in enumerate(files)
        try 
            push!(rows, compute_stats(f))
        catch
            failed[i] = true
        end
    end

    # group rows and files maintain the same ordering
    group = reduce(vcat, rows)
    files = files[.~failed]
    
    if length(files) == 0
        @warn "There are no valid files for $(modelname)/$(dataset)/seed=$(seed)"
        return
    end
    
    # select best based on criterion
    for criterion in [:val_auc, :val_tpr5, :val_pat_10]
        for select_top in [5, 10] # 0 could be automatic heuristic
            best = sortperm(group, order(criterion, rev=true))
            top = best[1:select_top]
            results = load.(files[top])

            scores, ensemble = _init_ensemble(results)
            for method in [:max, :mean, :wsum]
                ensemble = aggregate_score!(scores, deepcopy(ensemble), method=method)
                parameters = (modelname=modelname, criterion=criterion, size=select_top, method=method)

                savepath = replace(directory, "experiments" => "experiments_ensembles")
                tagsave(joinpath(savepath, savename("ensemble", parameters, "bson")), ensemble, safe = true)
            end            
        end
    end
end


const SPLITS = ["tr", "tst", "val"]

function _init_ensemble(results)
    ensemble = Dict{Symbol, Any}()      # new ensemble experiment dictionary
    r = first(results)                  # reference dictionary

    # add anomaly class
    for key in vcat([:dataset, :modelname, :seed, :model], _prefix_symbol.(SPLITS, :labels))
        ensemble[key] = r[key]
    end
    ensemble[:ensemble_parameters] = [r[:parameters] for r in results]
    
    scores = Dict()
    for key in _prefix_symbol.(SPLITS, :scores)
        kscores = similar(r[key], (length(r[key]), length(results)))
        for (i, rr) in enumerate(values(results))
            kscores[:,i] .= rr[key]
        end
        scores[key] = kscores
    end    
    scores, ensemble
end

# should probably go thorough all the methods while having the ensemble dictionary initialized
function aggregate_score(scores, ensemble, weights=ones(length(results))./length(results); 
                            method=:max)
    for (k, s) in scores
        if method == :max
            ensemble[k] = maximum(s, dims=2)[:]
        elseif method == :mean
            ensemble[k] = mean(s, dims=2)[:]
        elseif methods == :wsum
            ensemble[k] = sum(weights .* s, dims=2)[:]
        else
            error("Unsupported ensemble aggregation.")
        end
    end
    ensemble
end

