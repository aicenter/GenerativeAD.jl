uci_datasets = ["abalone", "blood-transfusion", "breast-cancer-wisconsin", "breast-tissue", 
	"cardiotocography", "ecoli", "glass", "haberman", "ionosphere", "iris", "isolet", 
	"letter-recognition", "libras", "magic-telescope", "miniboone", "multiple-features", 
	"page-blocks", "parkinsons", "pendigits", "pima-indians", "sonar", "spect-heart", "statlog-satimage", 
	"statlog-segment", "statlog-shuttle", "statlog-vehicle", "synthetic-control-chart", 
	"wall-following-robot", "waveform-1", "waveform-2", "wine", "yeast"]
	# "gisette", "madelon"] - no easy anomalies + very large in size
	# "vertebral-column"] - no easy and medium anomalies

"""
	load_uci_data(dataset::String)

Loads basic UCI data. For a list of available datasets, check `GenerativeAD.Datasets.uci_datasets`.
"""
function load_uci_data(dataset::String)
	# I have opted for the original Loda datasets, use of multiclass problems in all vs one case
	# does not necessarily represent a good anomaly detection scenario
	data, _, _ = UCI.get_loda_data(dataset)
	# return only easy and medium anomalies
	UCI.normalize(data.normal, hcat(data.easy, data.medium)) # data (standardized)
end

other_datasets = ["annthyroid", "arrhythmia", "htru2", "kdd99", "kdd99_small", "spambase", "mammography", "har", "seismic"]

"""
	init_tabular()

If not present, downloads tabular datasets.
"""
init_tabular() = map(d->(@datadep_str d), filter(x->x != "kdd99_small", other_datasets))

"""
	load_other_dataset(dataset[; standardize=false])

Load a dataset that is not a part of the UCI.jl package. For a list of available datasets, check
`GenerativeAD.Datasets.other_datasets`.
"""
function load_other_data(dataset; standardize = false)
	load_f = eval(Symbol("load_"*dataset))
	data_normal, data_anomalous = load_f()
	if standardize
		return UCI.normalize(data_normal, data_anomalous) 
	else
		return data_normal, data_anomalous
	end
end

"""
	load_arrhythmia()
"""
function load_arrhythmia()
	data_path = datadep"arrhythmia"
	f = joinpath(data_path, readdir(data_path)[1])
	raw_data = readdlm(f, ',')
	
	# trow away features containing missing values
	proc_data = raw_data[:,filter(x->!(x in [11,12,13,14]), 1:size(raw_data,2))]
	# now throw away one observation with a missing value
	proc_data = proc_data[filter(x->!(x in [5]), 1:size(proc_data,1)), :]
	data = Array{Float32,2}(transpose(proc_data[:,1:end-1]))
	labels = proc_data[:,end]

	data[:, labels.==1], data[:, labels.!=1]
end

"""
	load_annthyroid()
"""
function load_annthyroid()
	data_path = datadep"annthyroid"
	fs = joinpath.(data_path, readdir(data_path))
	raw_data = readdlm.(fs, Float32)
	raw_data = Array(transpose(vcat(raw_data...)))
	labels = raw_data[end,:]
	data = raw_data[1:end-1,:]
	data[:, labels.==3], data[:, labels.!=3]
end

"""
	load_htru2()
"""
function load_htru2()
	data_path = datadep"htru2"
	f = joinpath(data_path, "HTRU_2.csv")
	raw_data = CSV.read(f, DataFrame, header = false)
	data = Array{Float32,2}(transpose(Array(raw_data[:,1:end-1])))
	labels = raw_data[:,end]
	data[:, labels.==0], data[:, labels.==1]
end

function _load_kdd99(f::String)
	raw_data = readdlm(f, ',')
	M,N = size(raw_data)
	labels = raw_data[:,end]
	# the 2nd, 3rd and 4th columns contain nonnumericalvalues and must be one hot encoded
	# 2nd col = 3 unique vals
	# 3 = 66
	# 4 = 11
	unss = [unique(raw_data[:,2]), unique(raw_data[:,3]), unique(raw_data[:,4])] 
	ls = length.(unss)
	data = zeros(Float32,N+sum(ls)-4, M) # this is the final output array
	# copy first col
	data[1, :] = raw_data[:,1]
	# onehot encode the 3 following
	start_ind = 1
	for (uns,l,j) in zip(unss,ls,2:4)
		for i in 1:M
			data[(start_ind+1):(start_ind+l),i] = Flux.onehot(raw_data[i,j], uns)
		end
		start_ind = start_ind + l
	end
	# now copy the rest
	data[(sum(ls)+2):end,:] = transpose(raw_data[:,5:end-1])

	data[:, labels.=="normal."], data[:, labels.!="normal."]
end

"""
	load_kdd99_small()
"""
function load_kdd99_small()
	data_path = datadep"kdd99"
	f = joinpath(data_path, "kddcup.data_10_percent")
	_load_kdd99(f)
end

"""
	load_kdd99()
"""
function load_kdd99()
	data_path = datadep"kdd99"
	f = joinpath(data_path, "kddcup.data")
	_load_kdd99(f)
end

"""
	load_spambase()
"""
function load_spambase()
	data_path = datadep"spambase"
	f = joinpath(data_path, "spambase.data")
	raw_data = readdlm(f, ',', Float32)
	data = Array{Float32,2}(transpose(Array(raw_data[:,1:end-1])))
	labels = raw_data[:,end]
	data[:, labels.==0], data[:, labels.==1]
end

"""
	load_mammography()
"""
function load_mammography()
	data_path = datadep"mammography"
	f = joinpath(data_path, readdir(data_path)[1])
	raw_data = CSV.read(f, DataFrame)
	labels = parse.(Int,replace.(raw_data[!,:class], "'" => ""))
	data = Array{Float32,2}(transpose(Array(raw_data[:,1:end-1])))
	data[:, labels.==-1], data[:, labels.==1]
end

"""
	load_har()
"""
function load_har()
	data_path = datadep"har"
	X_test = readdlm(joinpath(data_path, "UCI HAR Dataset/test/X_test.txt"), Float32)
	X_train = readdlm(joinpath(data_path, "UCI HAR Dataset/train/X_train.txt"), Float32)
	y_test = readdlm(joinpath(data_path, "UCI HAR Dataset/test/y_test.txt"), Float32)
	y_train = readdlm(joinpath(data_path, "UCI HAR Dataset/train/y_train.txt"), Float32)

	data = Array(transpose(vcat(X_test, X_train)))
	labels = vec(vcat(y_test, y_train))
	data[:, labels.!=6], data[:, labels.==6] # other x laying
end

"""
	load_seismic()
"""
function load_seismic()
	data_path = datadep"seismic"
	f = joinpath(data_path, "seismic-bumps.arff")
	M,N = (2584,19)
	raw_data = Array{Any,2}(zeros(M,N))
	open(f) do file
		for (i,ln) in enumerate(eachline(file))
			(i > 154) ? (raw_data[i-154,:] = split(ln, ",")) : nothing
		end
	end
	labels = parse.(Int,raw_data[:,end])

	# the 1st, 2nd, 3rd and 8th columns contain nonnumericalvalues and must be one hot encoded
	unss = [unique(raw_data[:,1]), unique(raw_data[:,2]), unique(raw_data[:,3]), unique(raw_data[:,8])] 
	ls = length.(unss)
	data = zeros(Float32,N+sum(ls)-5, M) # this is the final output array
	start_ind = 0
	for (uns,l,j) in zip(unss[1:3],ls[1:3],1:3)
		for i in 1:M
			data[(start_ind+1):(start_ind+l),i] = Flux.onehot(raw_data[i,j], uns)
		end
		start_ind = start_ind + l
	end
	
	# copy columns 4:7
	data[start_ind+1:start_ind+4,:] = transpose(parse.(Float32,raw_data[:,4:7]))
	
	# encode and fill col 8
	start_ind = start_ind+5
	uns = unss[4]
	l = ls[4]
	for i in 1:M
		data[start_ind:start_ind+l-1,i] = Flux.onehot(raw_data[i,8], uns)
	end
	
	# now copy the rest
	data[(sum(ls)+5):end,:] = transpose(parse.(Float32, raw_data[:,9:end-1]))

	data[:,labels.==0], data[:,labels.==1]
end

