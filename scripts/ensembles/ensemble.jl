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
    "anomaly_classes"
        arg_type = Int
        default = 10
        help = "Number of anomaly_classes to go through with each dataset."
end

function ensemble_experiment(eval_directory, exp_directory, out_directory)
    eval_files = isdir(eval_directory) ? readdir(eval_directory, join=true) : String[]
    if length(eval_files) == 0
        @warn "There are no valid files in $eval_directory"
        return
    end

    # load the dataframe
    df = reduce(vcat, map(f -> load(f)[:df], eval_files))

    # select best based on criterion
    for criterion in [:val_auc, :val_tpr_5, :val_pat_10]
        best = sortperm(df, order(criterion, rev=true))
        for select_top in [5, 10] # 0 ... automatic ensemble size
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
            files_to_load = joinpath.(exp_directory, exp_files)
            results = if all(isfile.(files_to_load))
                load.(files_to_load)
            else
                @warn "Discrepancy between $exp_directory and $eval_directory"
                return
            end

            scores, ensemble = _init_ensemble(results)
            ensemble[:ensemble_files] = exp_files
            for method in [:max, :mean]
                for ignore in [true, false]
                    eagg, fit_t, _, _, _ = @timed aggregate_score!(deepcopy(ensemble), scores;
                                        method=method, ignore_nan=ignore)
                    parameters = (
                        modelname=ensemble[:modelname], 
                        criterion=criterion, 
                        size=select_top, 
                        method=method,
                        ignore_nan=ignore)
                    eagg[:parameters] = parameters
                    eagg[:fit_t] = fit_t
                    eagg[:tr_eval_t] = 0.0
                    eagg[:tst_eval_t] = 0.0
                    eagg[:val_eval_t] = 0.0

                    savef = joinpath(out_directory, savename("ensemble", parameters, "bson"))
                    @info "Saving ensemble experiment to $savef"
                    tagsave(savef, eagg, safe = true)
                end
            end            
        end
    end
end


const SPLITS = ["tr", "tst", "val"]

function _init_ensemble(results)
    ensemble = Dict{Symbol, Any}()      # new ensemble experiment dictionary
    r = first(results)                  # reference dictionary

    for key in vcat([:dataset, :modelname, :seed, :model], _prefix_symbol.(SPLITS, :labels))
        ensemble[key] = r[key]
    end
    # add anomaly class if present
    if Symbol("anomaly_class") in keys(r)
        ensemble[:anomaly_class] = r[:anomaly_class]
	elseif Symbol("ac") in keys(r)
        ensemble[:anomaly_class] = r[:ac]
	end

    ensemble[:ensemble_phash] = [hash(r[:parameters]) for r in results]

    # sum training times
    ensemble[:ensemble_fit_t] = sum([r[:fit_t] for r in results])
    ensemble[:ensemble_eval_t] = sum([r[:tr_eval_t] + r[:tst_eval_t] + r[:val_eval_t] for r in results])

    scores = Dict()
    for key in _prefix_symbol.(SPLITS, :scores)
        kscores = similar(r[key], (length(r[key]), length(results)))
        for (i, rr) in enumerate(values(results))
            kscores[:,i] .= rr[key][:]
        end
        scores[key] = kscores
    end    
    scores, ensemble
end

function aggregate_score!(ensemble, scores, weights=nothing; 
                            method=:max, ignore_nan=true)
    for (k, s) in scores
        if method == :max
            mask_nan_max = (x) -> (isnan(x) ? -Inf : x)
            ensemble[k] = ignore_nan ? 
                        maximum(mask_nan_max, s, dims=2)[:] : 
                        maximum(s, dims=2)[:]
        elseif method == :mean
            ensemble[k] = ignore_nan ? 
                        [mean(filter(x -> !isnan(x), s[i,:])) for i in 1:size(s, 1)] :
                        mean(s, dims=2)[:]
        elseif method == :wsum
            ensemble[k] = sum(weights' .* s, dims=2)[:]
        else
            error("Unsupported ensemble aggregation.")
        end
    end
    ensemble
end

function main(args)
    @unpack modelname, dataset, dataset_type, max_seed, anomaly_classes = args
    if dataset_type == "tabular"
        for s in 1:max_seed
            eval_directory = datadir("evaluation/$(dataset_type)/$(modelname)/$(dataset)/seed=$(s)/")
            exp_directory = datadir("experiments/$(dataset_type)/$(modelname)/$(dataset)/seed=$(s)/")
            # for now the ensemble outputs are kept in separate directory
            out_directory = datadir("experiments_ensembles/$(dataset_type)/$(modelname)/$(dataset)/seed=$(s)/")
            
            # run experiment
            @info "Generating ensemble scores of $(modelname) on $(dataset):$(s)"
            ensemble_experiment(eval_directory, exp_directory, out_directory)
        end
    elseif dataset_type == "images"
        for s in 1:max_seed
            for ac in 1:anomaly_classes
                eval_directory = datadir("evaluation/$(dataset_type)/$(modelname)/$(dataset)/ac=$(ac)/seed=$(s)/")
                exp_directory = datadir("experiments/$(dataset_type)/$(modelname)/$(dataset)/ac=$(ac)/seed=$(s)/")
                # for now the ensemble outputs are kept in separate directory
                out_directory = datadir("experiments_ensembles/$(dataset_type)/$(modelname)/$(dataset)/ac=$(ac)/seed=$(s)/")
                
                # run experiment
                @info "Generating ensemble scores of $(modelname) on $(dataset):$(ac):$(s)"
                ensemble_experiment(eval_directory, exp_directory, out_directory)
            end
        end
    else
        @error "Unsupported dataset type."
    end
end

main(parse_args(ARGS, s))