using ArgParse
using DrWatson
@quickactivate
using BSON
using Random
using FileIO
using Statistics
using DataFrames
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
criterion = :val_auc
select_top = 10
method = :max
###

# this has to change for images
function ensemble_experiment(modelname, dataset, dataset_type, seed; kwargs...)
    # add some info log

    eval_directory = datadir("evaluation/$(dataset_type)/$(modelname)/$(dataset)/seed=$(seed)/")
    exp_directory = datadir("experiments/$(dataset_type)/$(modelname)/$(dataset)/seed=$(seed)/")

    eval_files = readdir(eval_directory, join=true)
    if length(eval_files) == 0
        @warn "There are no valid files for $(modelname)/$(dataset)/seed=$(seed)"
        return
    end

    # load the dataframe
    df = reduce(vcat, map(f -> load(f)[:df], eval_files))

    # select best based on criterion
    for criterion in [:val_auc, :val_tpr_5, :val_pat_10]
        for select_top in [0, 5, 10] # 0 ... automatic ensemble size
            best = sortperm(df, order(criterion, rev=true))
            
            if (select_top > 0) && (length(best) > select_top)
                top = best[1:select_top]
            elseif (length(best) <= select_top)
                top = best
            else
                @warn "Automatic ensemble size is not yet implemented."
                continue
            end

            # get rid of the "eval_" prefix
            exp_files = replace.(basename.(eval_files[top]), "eval_" => "")
            # this assumes that evaluation and experiments files are in sync
            # more precisely for every evaluation file there exist it's exp.
            results = load.(joinpath.(exp_directory, exp_files))

            scores, ensemble = _init_ensemble(results)
            for method in [:max, :mean, :wsum]
                eagg = aggregate_score!(deepcopy(ensemble), scores, method=method)
                parameters = (modelname=modelname, criterion=criterion, size=select_top, method=method)

                savepath = replace(exp_directory, "experiments" => "experiments_ensembles")
                savef = joinpath(savepath, savename("ensemble", parameters, "bson"))
                @info "Saving ensemble experiment to $savef"
                tagsave(savef, eagg, safe = true)
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

function aggregate_score!(ensemble, scores, weights=nothing; 
                            method=:max)
    for (k, s) in scores
        if method == :max
            ensemble[k] = maximum(s, dims=2)[:]
        elseif method == :mean
            ensemble[k] = mean(s, dims=2)[:]
        elseif method == :wsum
            # placeholder
            ss = size(s, 2)
            weights = ones(eltype(s), (1, ss))./ss
            # placeholder
            ensemble[k] = sum(weights .* s, dims=2)[:]
        else
            error("Unsupported ensemble aggregation.")
        end
    end
    ensemble
end

