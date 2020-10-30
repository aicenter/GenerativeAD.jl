using DrWatson
@quickactivate
using BSON
using DataFrames
using ValueHistories
using Statistics
using CSV


function fix_info_name(name, score="reconstruction")
	spl = split(name, "lr=0_")
	name = (length(spl)==2) ? spl[1]*"lr=0.0001_"*spl[2] : name
	params = DrWatson.parse_savename("_"*name)[2]
	params = merge(params, Dict("score"=>score))
	name = DrWatson.savename(param, digits=5)*".bson"
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

function create_df(models; images::Bool=true)

	if images
		df = DataFrame(
			path = String[], #path to model / encodings
			params = String[],
			dataset = String[], 
			ac = Int64[], 
			seed = Int64[], 
			loss_val= Float32[]
		)
	else
		df = DataFrame(
			path = String[], #path to model / encodings
			params = String[],
			dataset = String[], 
			seed = Int64[], 
			loss_val= Float32[]
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
						minimum(get(info[:history][:validation_likelihood])[2]) # early stopping in min
					]
				else
					update = [
						joinpath(root,mod), 
						string(info[:parameters]),
						info[:dataset], 
						info[:seed],
						minimum(get(info[:history][:validation_likelihood])[2]) # early stopping in min
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

#path = "/home/skvarvit/generativead/GenerativeAD.jl/data/experiments/images/vae/"
#models = models_and_params(path)
#df = create_df(models, images=true)
#CSV.write(datadir("vae_tab.csv"), df)

encoders = ["vae","wae", "wae_vamp"]
for type in ["images", "tabular"]
	for encoder in encoders
		models = models_and_params(datadir("experiments/$(type)/$(encoder)"))
		df = create_df(models, images=(type=="images"))
		CSV.write(datadir("$(encoder)_$(type)_tab.csv"), df)
		println("$(encoder)-$(type) ... done")
	end
end
