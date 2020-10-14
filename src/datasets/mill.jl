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
