uci_datasets = ["abalone", "blood-transfusion", "breast-cancer-wisconsin", "breast-tissue", 
	"cardiotocography", "ecoli", "gisette", "glass", "haberman", "ionosphere", "iris", "isolet", 
	"letter-recognition", "libras", "madelon", "magic-telescope", "miniboone", "multiple-features", 
	"page-blocks", "parkinsons", "pendigits", "pima-indians", "sonar", "spect-heart", "statlog-satimage", 
	"statlog-segment", "statlog-shuttle", "statlog-vehicle", "synthetic-control-chart", 
	"wall-following-robot", "waveform-1", "waveform-2", "wine", "yeast"]
	#"vertebral-column"] - no easy and medium anomalies

"""
	load_data(dataset::String, ratios=(0.6,0.2,0.2); seed=nothing, contamination::Real=0.0)

Returns 3 tuples of (data, labels) representing train/validation/test part. Arguments are the splitting
ratios for normal data, seed and training data contamination.
"""
function load_data(dataset::String, ratios=(0.6,0.2,0.2); seed=nothing, contamination::Real=0.0)
	# extract data and labels
	if dataset in uci_datasets # UCI Loda data, standardized
		data_normal, data_anomalous = load_uci_data(dataset)
	else
		error("Dataset not known, either not implemented or misspeled.")
		# TODO add the rest
	end

	# now do the train/validation/test split
	return train_val_test_split(data_normal, data_anomalous, ratios; seed=seed, contamination=contamination)
end

"""
	load_uci_data(dataset::String)

Loads basic UCI data.
"""
function load_uci_data(dataset::String)
	# I have opted for the original Loda datasets, use of multiclass problems in all vs one case
	# does not necessarily represent a good anomaly detection scenario
	data, _, _ = UCI.get_loda_data(dataset) 
	# return only easy and medium anomalies
	return UCI.normalize(data.normal, hcat(data.easy, data.medium)) # data (standardized)
end

"""
    train_val_test_inds(indices, ratios=(0.6,0.2,0.2); seed=nothing)

Split indices.
"""
function train_val_test_inds(indices, ratios=(0.6,0.2,0.2); seed=nothing)
    (sum(ratios) == 1 && length(ratios) == 3) ? nothing : 
    	error("ratios must be a vector of length 3 that sums up to 1")
    
    # set seed
    (seed == nothing) ? nothing : Random.seed!(seed)
    
    # set number of samples in individual subsets
    n = length(indices)
    ns = cumsum([x for x in floor.(Int, n .* ratios)])
    
    # scramble indices
    _indices = sample(indices, n, replace=false)
    
    # restart seed
    (seed == nothing) ? nothing : Random.seed!()
    
    # return the sets of indices
    return _indices[1:ns[1]], _indices[ns[1]+1:ns[2]], _indices[ns[2]+1:ns[3]]
end

"""
	train_val_test_split(data_normal, data_anomalous, ratios=(0.6,0.2,0.2); seed=nothing,
	    	contamination::Real=0.0)

Split data.
"""
function train_val_test_split(data_normal, data_anomalous, ratios=(0.6,0.2,0.2); seed=nothing,
	    	contamination::Real=0.0)
	# split the normal data, add some anomalies to the train set and divide 
	# the rest between validation and test
	(0 <= contamination <= 1) ? nothing : error("contamination must be in the interval [0,1]")
	nd = ndims(data_normal) # differentiate between 2D tabular and 4D image data

	# split normal indices
	indices = 1:size(data_normal, nd)
	split_inds = train_val_test_inds(indices, ratios; seed=seed)

	# select anomalous indices
	indices_anomalous = 1:size(data_anomalous, nd)
	vtr = (1 - contamination)/2 # validation/test ratio
	split_inds_anomalous = train_val_test_inds(indices_anomalous, (contamination, vtr, vtr); seed=seed)

	# this can be done universally - how?
	if nd == 2
		tr_n, val_n, tst_n = map(is -> data_normal[:,is], split_inds)
		tr_a, val_a, tst_a = map(is -> data_anomalous[:,is], split_inds_anomalous)
	elseif nd == 4
		tr_n, val_n, tst_n = map(is -> data_normal[:,:,:,is], split_inds)
		tr_a, val_a, tst_a = map(is -> data_anomalous[:,:,:,is], split_inds_anomalous)
	end

	# cat it together
	tr_x = cat(tr_n, tr_a, dims = nd)
	val_x = cat(val_n, val_a, dims = nd)
	tst_x = cat(tst_n, tst_a, dims = nd)
	
	# now create labels
	tr_y = vcat(zeros(size(tr_n, nd)), ones(size(tr_a,nd)))
	val_y = vcat(zeros(size(val_n, nd)), ones(size(val_a,nd)))
	tst_y = vcat(zeros(size(tst_n, nd)), ones(size(tst_a,nd)))

	return (tr_x, tr_y), (val_x, val_y), (tst_x, tst_y)
end
