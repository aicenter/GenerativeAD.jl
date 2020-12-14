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
function __init__()
	register(
		DataDep(
			"annthyroid",
			"""
			Dataset: ANNThyroid
			Authors: Ross Quinlan
			Website: https://archive.ics.uci.edu/ml/datasets/Thyroid+Disease
			
			[Quinlan, 1987]
				Quinlan, John Ross, et al. 
				"Inductive knowledge acquisition: a case study." 
				Proceedings of the Second Australian Conference on Applications of expert systems. 1987.
			
			The original thyroid disease (ann-thyroid) dataset from UCI machine learning repository is 
			a classification dataset, which is suited for training ANNs. It has 3772 training instances 
			and 3428 testing instances. It has 15 categorical and 6 real attributes. The problem is to 
			determine whether a patient referred to the clinic is hypothyroid. Therefore three classes 
			are built: normal (not hypothyroid), hyperfunction and subnormal functioning.
			""",
			[
				"https://archive.ics.uci.edu/ml/machine-learning-databases/thyroid-disease/ann-train.data",
				"https://archive.ics.uci.edu/ml/machine-learning-databases/thyroid-disease/ann-test.data"
			],
			"fbecd00c2ab44e0c95fb135e9c2f39d597fb8074dfaa8581ac2f889c927d40ad"
		))
	# this ensures eager loading (upon first usage of the package)
	# datadep"annthyroid"
	# we dont do that no more

	register(
		DataDep(
			"arrhythmia",
			"""
			Dataset: Arrhythmia
			Authors: H. Altay Guvenir
			Website: https://archive.ics.uci.edu/ml/datasets/Arrhythmia
			
			[Guvenir, 1997]
				Guvenir, H. Altay, et al. 
				"A supervised machine learning algorithm for arrhythmia analysis." 
				Computers in Cardiology 1997. IEEE, 1997.
			
			The aim is to distinguish between the presence and absence of cardiac arrhythmia and to classify 
			it in one of the 16 groups. Class 01 refers to 'normal' ECG classes 02 to 15 refers to different 
			classes of arrhythmia and class 16 refers to the rest of unclassified ones. For the time being, 
			there exists a computer program that makes such a classification. However there are differences 
			between the cardiolog's and the programs classification. Taking the cardiolog's as a gold standard 
			we aim to minimise this difference by means of machine learning tools.
			""",
			"https://archive.ics.uci.edu/ml/machine-learning-databases/arrhythmia/arrhythmia.data",
			"a7f0f4ca289a4c58b5ed85a9adb793189acd38425828ce3dfbb70adb45f30169"
		))

	register(
		DataDep(
			"htru2",
			"""
			Dataset: HTRU2
			Authors: Dr. Robert Lyon
			Website: https://archive.ics.uci.edu/ml/datasets/HTRU2
			
			[Keith, 2010]
				M. J. Keith et al.
				"The High Time Resolution Universe Pulsar Survey - I. System Configuration and Initial Discoveries." 
				onthly Notices of the Royal Astronomical Society, 2010.
			
			HTRU2 is a data set which describes a sample of pulsar candidates collected during 
			the High Time Resolution Universe Survey (South).
			""",
			"https://archive.ics.uci.edu/ml/machine-learning-databases/00372/HTRU2.zip",
			"ba442c076dd22a8952700f26e38499fc1806037dcf7bea0e125e6bfba393f379",
			post_fetch_method = unpack
		))

	register(
		DataDep(
			"kdd99",
			"""
			Dataset: KDD Cup 1999
			Authors: Stolfo et al.
			Website: http://kdd.ics.uci.edu/databases/kddcup99/kddcup99.html
			
			[Stolfo, 2000]
				 S. J. Stolfo, W. Fan, W. Lee, A. Prodromidis, and P. K. Chan. 
				"Costbased modeling for fraud and intrusion detection: Results from the jam project." 
				discex, 2000.

			This is the data set used for The Third International Knowledge Discovery and Data Mining Tools
			Competition, which was held in conjunction with KDD-99 The Fifth International Conference on 
			Knowledge Discovery and Data Mining. The competition task was to build a network intrusion 
			detector, a predictive model capable of distinguishing between ``bad'' connections, called 
			intrusions or attacks, and ``good'' normal connections. This database contains a standard set 
			of data to be audited, which includes a wide variety of intrusions simulated in a military 
			network environment.

			!!!!!!!!!!!!!!!!!!!!

			WARNING --- for some users, the automatic unpacking may fail. Run this instead:

			mkdir ~/.julia/datadeps/kdd99
			cd ~/.julia/datadeps/kdd99
			wget http://kdd.ics.uci.edu/databases/kddcup99/kddcup.data.gz
			wget http://kdd.ics.uci.edu/databases/kddcup99/kddcup.data_10_percent.gz
			gzip -d kddcup.data.gz
			gzip -d kddcup.data_10_percent.gz

			!!!!!!!!!!!!!!!!!!!!
			""",
			["http://kdd.ics.uci.edu/databases/kddcup99/kddcup.data.gz", 
			"http://kdd.ics.uci.edu/databases/kddcup99/kddcup.data_10_percent.gz"],
			"bb29388a787b1cea818a6fd427d14ad45b556176b5fbf13257ca33c1d2dad7f3",
			post_fetch_method = unpack
		))

	register(
		DataDep(
			"spambase",
			"""
			Dataset: Spambase
			Authors: Mark Hopkins, Erik Reeber, George Forman, Jaap Suermondt
			Website: https://archive.ics.uci.edu/ml/datasets/spambase
			
			Our collection of spam e-mails came from our postmaster and individuals who had filed spam. 
			Our collection of non-spam e-mails came from filed work and personal e-mails, and hence 
			the word 'george' and the area code '650' are indicators of non-spam. These are useful when 
			constructing a personalized spam filter. One would either have to blind such non-spam indicators
			or get a very wide collection of non-spam to generate a general purpose spam filter.
			""",
			"https://archive.ics.uci.edu/ml/machine-learning-databases/spambase/spambase.data",
			"b1ef93de71f97714d3d7d4f58fc9f718da7bbc8ac8a150eff2778616a8097b12"
		))

	register(
		DataDep(
			"mammography",
			"""
			Dataset: Mammography
			Authors: Tobias Kuehn
			Website: https://www.openml.org/d/310
			
			Mammography dataset.
			""",
			"https://www.openml.org/data/get_csv/52214/phpn1jVwe",
			"d39362668aa89b2b48aa0e6c2d12277850ace4c390e15b89862b7717f1525f0c"
			))

	register(
		DataDep(
			"har",
			"""
			Dataset: Human Activity Recognition
			Authors: Jorge L. Reyes
			Website: https://archive.ics.uci.edu/ml/datasets/human+activity+recognition+using+smartphones
			
			Human Activity Recognition dataset. Normal class = "WALKING", anomalous class = other.
			""",
			"https://archive.ics.uci.edu/ml/machine-learning-databases/00240/UCI%20HAR%20Dataset.zip",
			"2045e435c955214b38145fb5fa00776c72814f01b203fec405152dac7d5bfeb0",
			post_fetch_method = unpack
			))

	register(
		DataDep(
			"seismic",
			"""
			Dataset: Seismic activity
			Authors: Marek Sikora
			Website: https://archive.ics.uci.edu/ml/datasets/seismic-bumps
			
			Seismic bumps from mining activity.
			""",
			"https://archive.ics.uci.edu/ml/machine-learning-databases/00266/seismic-bumps.arff",
			"aabe512fab65b36d1dfb462650b75cfd8d99d8cc2723e8ecb4e6f5e1caccd5a7",
			))
end

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

