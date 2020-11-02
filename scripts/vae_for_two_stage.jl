using DrWatson
@quickactivate
using BSON
using DataFrames
using ValueHistories
using Statistics
using EvalMetrics
using CSV


function compute_score(info, score="AUC")
	(scores, labels) = (info[:val_scores], info[:val_labels])
	if score == "AUC"
		roc = EvalMetrics.roccurve(labels, scores)
		auc = EvalMetrics.auc_trapezoidal(roc...)
		return auc
	elseif score == "AUPRC"
		prc = EvalMetrics.prcurve(labels, scores)
		auprc = EvalMetrics.auc_trapezoidal(prc...)
		return auprc
	elseif score == "TPR@5"
		t5 = EvalMetrics.threshold_at_fpr(labels, scores, 0.05)
		cm5 = ConfusionMatrix(labels, scores, t5)
		tpr5 = EvalMetrics.true_positive_rate(cm5)
		return tpr5
	elseif score == "F1@5"
		t5 = EvalMetrics.threshold_at_fpr(labels, scores, 0.05)
		cm5 = ConfusionMatrix(labels, scores, t5)
		f5 = EvalMetrics.f1_score(cm5)
		return f5
	else 
		@error "unknown score"
	end 
end

function fix_info_name(name, score="reconstruction")
	spl = split(name, "lr=0_")
	name = (length(spl)==2) ? spl[1]*"lr=0.0001_"*spl[2] : name
	params = DrWatson.parse_savename("_"*name)[2]
	params = merge(params, Dict("score"=>score))
	name = DrWatson.savename(params, digits=5)*".bson"
	return name
end

function models_and_params(path_to_model)
	directories = []
	models = []
	
	for (root, dirs, files) in walkdir(path_to_model)
		for file in files
			push!(directories, root)
		end
	end
	
	for dir in unique(directories)
		fs = filter(x->(startswith(x, "model")), readdir(dir))
		info = map(x->x[7:end], fs)
		info = fix_info_name.(info) # model_name -> info name
		#par = map(x -> DrWatson.parse_savename("_"*x)[2], info)
		!isempty(fs) ? push!(models, (dir, fs, info)) : nothing
	end
	return models
end

function create_df(models; score::String="val_loss", images::Bool=true)

	if images
		df = DataFrame(
			path = String[], #path to model / encodings
			params = String[],
			dataset = String[], 
			ac = Int64[], 
			seed = Int64[], 
			criterion = Float32[]
		)
	else
		df = DataFrame(
			path = String[], #path to model / encodings
			params = String[],
			dataset = String[], 
			seed = Int64[], 
			criterion = Float32[]
		)
	end

	i = 1

	for model in models
		(root, mod_path, infos) = model
		for (mod,info) in zip(mod_path, infos)
			try
				path = joinpath(root, info)
				info = BSON.load(path)
				if images
					update = [
						joinpath(root,mod), 
						string(info[:parameters]),
						info[:dataset], 
						info[:anomaly_class], 
						info[:seed],
						(score=="val_loss") ? minimum(get(info[:history][:validation_likelihood])[2]) : compute_score(info, score)
					]
				else
					update = [
						joinpath(root,mod), 
						string(info[:parameters]),
						info[:dataset], 
						info[:seed],
						(score=="val_loss") ? minimum(get(info[:history][:validation_likelihood])[2]) : compute_score(info, score)
					]
				end
				push!(df, update)
			catch e
				println("info not found #$(i)")
				i += 1
			end
		end
	end
	return df
end

function return_best_n(df, rev=false, n=10)
    #df = df[df.dataset .== dataset, :]
    dff = sort(by(df, [:params, :dataset], :criterion => mean), :criterion_mean, rev=rev)
	df_top = []
	for dataset in unique(dff.dataset)
		top = first(dff[dff.dataset .== dataset, :], n)
		for i=1:n
			tmp = df[df.params .== top.params[i],:]
			ind = i.*ones(size(tmp,1))
			push!(df_top, hcat(tmp, DataFrame(ind=ind)))
		end
	end
    return vcat(df_top...)
end

#path = "/home/skvarvit/generativead/GenerativeAD.jl/data/experiments/images/vae/"
#models = models_and_params(path)
#df = create_df(models, images=true)
#CSV.write(datadir("vae_tab.csv"), df)

encoders = ["vae","wae", "wae_vamp"]

for type in ["images", "tabular"]
	for encoder in encoders
		for score in ["val_loss", "AUC", "AUPRC", "TPR@5", "F1@5"]
			@info "working on $(type)-$(encoder)-$(score)"
			#models = models_and_params("/home/skvarvit/generativead/GenerativeAD.jl/data/experiments/$(type)/$(encoder)") #shortcut
			models = models_and_params(datadir("experiments/$(type)/$(encoder)"))
			df = create_df(models, score=score, images=(type=="images"))
			CSV.write(datadir("$(encoder)_$(score)_$(type)_tab.csv"), df)
			CSV.write(datadir("$(encoder)_$(score)_$(type)_best_tab.csv"), return_best_n(df, score!="val_loss", 10))
			println("$(encoder)-$(score)-$(type) ... done")
		end
	end
end
