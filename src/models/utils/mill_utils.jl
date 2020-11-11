# this contains stuff that is common for Mill evalution

"""
	raw_ll_score(model::AEModel, x)

Anomaly score based on the reconstruction probability of the data.
"""
function raw_ll_score(model::VAE,  b::BagNode) 
    # evaluate ll for each sample in a bag
    sci = (i)->(sum(GenerativeAD.Models.reconstruction_score_mean(model,b.data.data[:,b.bags[i]])))
    map(sci, 1:nobs(b))
end

