"""
	predict_anomaly_factor(tr_scores::AbstractArray{T,2}, x::Vector{T})

Predicts the most anomalous factor.
"""
function predict_anomaly_factor(tr_scores::AbstractArray{T,2}, x::Vector{T}) where T
	# cat all scores and the test one
	all_scores = hcat(tr_scores, x)
	# sort them
	sortinds = map(i->sortperm(all_scores[i,:]), 1:3)
	n = size(all_scores,2)
	# now find where the score x ranks the highest
	maxinds = map(inds->findfirst(inds .== n), sortinds)
	y_pred = argmax(maxinds)
end

"""
	predict_random(x::Vector)

Prediction of a random class.	
"""
predict_random(x::Vector) = sample(1:length(x))