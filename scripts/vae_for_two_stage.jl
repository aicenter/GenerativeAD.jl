using DrWatson
@quickactivate
using BSON
using DataFrames
using ValueHistories
using Statistics
using CSV

function fix_info_name(name)
	spl = split(name, "lr=0_")
	name = (length(spl)==2) ? spl[1]*"lr=0.0001_"*spl[2] : name
	spl = split(name, "zdim")
	name = spl[1]*"score=latent_zdim"*spl[2]
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

function create_df(models)
	df = DataFrame(
		path = String[], #path to model / encodings
		params = String[],
		dataset = String[], 
		ac = Int64[], 
		seed = Int64[], 
		loss_val= Float32[]
	)

	i = 1

	for model in models
		(root, mod, infos) = model
		for info in infos
			try
				path = joinpath(root, info)
				info = BSON.load(path)
				push!(
					df, 
					[
						mod, 
						string(info[:parameters]),
						info[:dataset], 
						info[:anomaly_class], 
						info[:seed],
						minimum(get(info[:history][:validation_likelihood])[2]) # early stopping in min
					]
				)
			catch e
				println("info not found #$(i)")
				i += 1
			end
		end
	end
	return df
end

path = "/home/skvarvit/generativead/GenerativeAD.jl/data/experiments/images/vae/"
models = models_and_params(path)
df = create_df(models)

#df_mean = by(df, [:parameters, :dataset], :loss_val => mean)

CSV.write(datadir("vae_tab.csv"), df)