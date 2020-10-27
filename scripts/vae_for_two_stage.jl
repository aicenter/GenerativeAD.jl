using DrWatson
@quickactivate
using BSON
using DataFrames
using ValueHistories
using LinearAlgebra
using CSV

function models_and_params(path_to_model)
	directories = []
	models = []
	
	for (root, dirs, files) in walkdir(path_to_model)
		for file in files
			push!(directories, root)
		end
	end
	
	for dir in unique(directories)
		fs = filter(x->!(occursin("_#", x)), readdir(dir))
		fs = filter(x->(startswith(x, "model")), fs)
		par = map(x -> DrWatson.parse_savename("_"*x)[2], fs)
		!isempty(fs) ? push!(models, (dir, fs, compare.(par))) : nothing
	end
	return models
end

path = "/home/skvarvit/generativead/GenerativeAD.jl/data/experiments/images/vae/"

models = models_and_params(path)


df = DataFrame(
	path = String[], 
	dataset = String[], 
	ac = Int64[], 
	seed = Int64[], 
	loss_val= Float32[]
)

for model in models
	(r, names, params) = model
    for (n,p) in zip(names, params)
        
        tr_info_name = n[7:end]
        info = BSON.load(joinpath(r, tr_info_name))

		push!(df, [joinpath(n, tr_info_name), 
					info[:dataset], 
					info[:anomaly_class], 
					info[:seed],
					minimum(get(info[:history][:validation_loss])[2]) # early stopping in min
					])

    end
end

CSV.write("vae_tab.csv", df)