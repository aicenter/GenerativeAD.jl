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

Loads basic UCI data.
"""
function load_uci_data(dataset::String)
	# I have opted for the original Loda datasets, use of multiclass problems in all vs one case
	# does not necessarily represent a good anomaly detection scenario
	data, _, _ = UCI.get_loda_data(dataset)
	# return only easy and medium anomalies
	UCI.normalize(data.normal, hcat(data.easy, data.medium)) # data (standardized)
end

other_datasets = ["annthyroid", "arrhythmia"]

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
    	),
    )
	# this ensures eager loading (upon first usage of the package)
    datadep"arrhythmia"
    datadep"annthyroid"
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
