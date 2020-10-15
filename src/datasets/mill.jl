mill_datasets = ["BrownCreeper","CorelBeach","Fox","Musk2","Mutagenesis2","Newsgroups2","Protein","UCSBBreastCancer","Web2","Web4","CorelAfrican","Elephant","Musk1","Mutagenesis1","Newsgroups1","Newsgroups3","Tiger","Web1","Web3","WinterWren"]



using DelimitedFiles, Mill, StatsBase

"""
    get_mill_datapath()

Get the absolute path of MIProblems data. !!! Hardcoded path to ../MIProblems
"""
get_mill_datapath() = joinpath(dirname(dirname(@__FILE__)), "../../MIProblems")

function seqids2bags(bagids)
	c = countmap(bagids)
	Mill.length2bags([c[i] for i in sort(collect(keys(c)))])
end


"""
	load_mill_data(dataset::String)

Loads basic MIProblems data. For a list of available datasets, check `GenerativeAD.Datasets.mill_datasets`.
"""
function load_mill_data(dataset::String)
	mdp = get_mill_datapath()
	x=readdlm("$mdp/$(problem)/data.csv",'\t',Float32)
	bagids = readdlm("$mdp/$(problem)/bagids.csv",'\t',Int)[:]
	y = readdlm("$mdp/$(problem)/labels.csv",'\t',Int)
	
	# plit to 0/1 classes
	c0 = y.==0
	c1 = y.==1
	bags0 = seqids2bags(bagids[c0[:]])
	bags1 = seqids2bags(bagids[c1[:]])

	# normalize to standard normal
	x.=UCI.normalize(x)
	# return normal and anomalous bags
	(normal = BagNode(ArrayNode(x[:,c0[:]]), bags0), anomaly = BagNode(ArrayNode(x[:,c1[:]]), bags1),)
end

import Base.length
Base.length(B::BagNode)=length(B.bags.bags)

"""
	train_val_test_split(data_normal, data_anomalous, ratios=(0.6,0.2,0.2); seed=nothing,
	    	contamination::Real=0.0)

Split Bag data.
"""
function train_val_test_split(data_normal::BagNode, data_anomalous::BagNode, ratios=(0.6,0.2,0.2); seed=nothing,
	    	contamination::Real=0.0)
	# split the normal data, add some anomalies to the train set and divide
	# the rest between validation and test
	(0 <= contamination <= 1) ? nothing : error("contamination must be in the interval [0,1]")
	nd = ndims(data_normal.data.data) # differentiate between 2D tabular and 4D image data

	# split normal indices
	indices = 1:length(data_normal.bags.bags)
	split_inds = train_val_test_inds(indices, ratios; seed=seed)

	# select anomalous indices
	indices_anomalous = 1:length(data_anomalous.bags.bags)
	vtr = (1 - contamination)/2 # validation/test ratio
	split_inds_anomalous = train_val_test_inds(indices_anomalous, (contamination, vtr, vtr); seed=seed)

	# this can be done universally - how?
	tr_n, val_n, tst_n = map(is -> data_normal[is], split_inds)
	tr_a, val_a, tst_a = map(is -> data_anomalous[is], split_inds_anomalous)

	# cat it together
	tr_x = cat(tr_n, tr_a, dims = nd)
	val_x = cat(val_n, val_a, dims = nd)
	tst_x = cat(tst_n, tst_a, dims = nd)

	# now create labels
	tr_y = vcat(zeros(Float32, length(tr_n)), ones(Float32, length(tr_a)))
	val_y = vcat(zeros(Float32, length(val_n)), ones(Float32, length(val_a)))
	tst_y = vcat(zeros(Float32, length(tst_n)), ones(Float32, length(tst_a)))

	(tr_x, tr_y), (val_x, val_y), (tst_x, tst_y)
end

