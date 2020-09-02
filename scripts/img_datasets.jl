using GenerativeAD

for dataset in GenerativeAD.mldatasets
	data = GenerativeAD.load_data(dataset, seed=1, anomaly_class_ind=1)
	n1 = sum(data[2][2]) + sum(data[3][2])
	n0 = size(data[1][1],4) + size(data[2][1],4) + size(data[3][1],4) - n1
	println("$dataset: ")
	println("      dims: $(size(data[1][1])[1:3])")
	println("      normal: $n0")
	println("      anomalous: $n1")
end
