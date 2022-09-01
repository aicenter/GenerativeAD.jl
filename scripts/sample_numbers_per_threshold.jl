using DrWatson
@quickactivate
using GenerativeAD
include("./supervised_comparison/utils.jl")
using CSV
using DataFrames
using GenerativeAD.Evaluation: _subsample_data

outf = datadir("evaluation/sample_numbers_per_threshold.csv")
datasets = ["CIFAR10", "SVHN2", "cocoplaces", "wildlife_MNIST"]
seed = 1
ac = 1
method = "leave-one-in"
ps = [NaN, 100.0, 50.0, 20.0, 10.0, 5.0, 2.0, 1.0, 0.5, 0.2, 0.1] ./ 100
cols = 	[:full, :n100, :n50, :n20, :n10, :n5, :n2, :n1, :n05, :n02, :n01]
outdf = DataFrame(
	:dataset => String[],
	:nn => Any[],
	:full => Any[],
	:n100 => Any[],
	:n50 => Any[],
	:n20 => Any[],
	:n10 => Any[],
	:n5 => Any[],
	:n2 => Any[],
	:n1 => Any[],
	:n05 => Any[],
	:n02 => Any[],
	:n01 => Any[],
	)

for dataset in datasets
	(tr_x, tr_y), (val_x, val_y), (tst_x, tst_y) = 
		GenerativeAD.load_data(dataset, seed=seed, anomaly_class_ind=ac, method=method);

	r = [dataset, Int(length(val_y)-sum(val_y))]
	for p in ps
		if isnan(p)
			push!(r, Int(sum(val_y)))
		else
			sub_x, sub_y = _subsample_data(p, 1.0, val_y, val_x)
			push!(r, Int(sum(sub_y)))
		end
	end
	push!(outdf, r)
end

categories = ["bottle",  "capsule",  "metal_nut", "pill", "transistor"]
for category in categories
	(tr_x, tr_y), (val_x, val_y), (tst_x, tst_y) = 
		GenerativeAD.load_data("MVTec-AD", seed=seed, category=category, img_size=128)
	r = [category, Int(length(val_y)-sum(val_y))]
	for p in ps
		if isnan(p)
			push!(r, Int(sum(val_y)))
		else
			try
				sub_x, sub_y = _subsample_data(p, 1.0, val_y, val_x)
				push!(r, Int(sum(sub_y)))
			catch e
				if isa(e, DomainError)
					push!(r, NaN)
				else
					rethrow(e)
				end
			end
		end
	end
	push!(outdf, r)
end

CSV.write(outf, outdf)
