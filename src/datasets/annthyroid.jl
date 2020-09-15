function __init__()
	register(DataDep(
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
    datadep"annthyroid"
end

function load_annthyroid()
	data_path = datadep"annthyroid"
	fs = joinpath.(data_path, readdir(data_path))
	raw_data = readdlm.(fs, Float32)
	raw_data = Array(transpose(vcat(raw_data...)))
	labels = raw_data[end,:]
	data = raw_data[1:end-1,:]
	data[:, labels.==3], data[:, labels.!=3]
end

