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


function logU_score(model::MillModel,  b::BagNode) 
    # evaluate ll for each sample in a bag
    sci = (i)->(sum(GenerativeAD.Models.reconstruction_score_mean(model.pf,b.data.data[:,b.bags[i]])))
	raw=map(sci, 1:nobs(b))
	ns = length.(b.bags)
	ll_pc = map(n->logpdf(model.pc,n), ns)

	ll_pc .+ raw .- ns.*model.U
end

