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

	register(
		DataDep(
			"MNIST-C",
			"""
			Dataset: MNIST-C
			Authors: Norman Mu, Justin Gilmer
			Website: https://github.com/google-research/mnist-c
			
			Corrupted MNIST dataset.
			""",
			"https://zenodo.org/record/3239543/files/mnist_c.zip?download=1",
			"af9ee8c6a815870c7fdde5af84c7bf8db0bcfa1f41056db83871037fba70e493",
			post_fetch_method = unpack
			))

end

