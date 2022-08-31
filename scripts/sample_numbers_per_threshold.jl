using DrWatson
@quickactivate
using GenerativeAD
include("./supervised_comparison/utils.jl")
using DataFrames
using GenerativeAD.Evaluation: _subsample_data

datasets = ["CIFAR10", "SVHN2", "cocoplaces", "wildlife_MNIST"]
seed = 1
ac = 1
method = "leave-one-in"
ps = [NaN, 100.0, 50.0, 20.0, 10.0, 5.0, 2.0, 1.0, 0.5, 0.2, 0.1] ./ 100
cols = 	[:full, :n100, :n50, :n20, :n10, :n5, :n2, :n1, :n05, :n02, :n01]
outdf = DataFrame(
	:dataset => String[],
	:nn => Int[],
	:full => Int[],
	:n100 => Int[],
	:n50 => Int[],
	:n20 => Int[],
	:n10 => Int[],
	:n5 => Int[],
	:n2 => Int[],
	:n1 => Int[],
	:n05 => Int[],
	:n02 => Int[],
	:n01 => Int[],
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

mvtecdf = DataFrame(
	:dataset => String[],
	:nn => Int[],
	:full => Int[],
	:n100 => Int[],
	:n50 => Int[],
	:n20 => Int[],
	:n10 => Int[],
	:n5 => Int[],
	:n2 => Int[],
	:n1 => Int[],
	:n05 => Int[],
	:n02 => Int[],
	:n01 => Int[],
	)

categories = ["bottle",  "capsule",  "metal_nut", "pill", "transistor"]
for cat in categories
	(tr_x, tr_y), (val_x, val_y), (tst_x, tst_y) = 
		GenerativeAD.load_data("MVTec-AD", seed=seed, category=cat, img_size=128)
	r = [cat, Int(length(val_y)-sum(val_y))]
	for p in ps
		if isnan(p)
			push!(r, Int(sum(val_y)))
		else
			try
				sub_x, sub_y = _subsample_data(p, 1.0, val_y, val_x)
				push!(r, Int(sum(sub_y)))
			catch e
				if isa(DomainError, e)
					push!(r, NaN)
				else
					rethrow(e)
				end
			end
		end
	end
	push!(outdf, r)
end
