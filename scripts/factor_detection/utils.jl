"""
	predict_ranked(tr_scores::AbstractArray{T,2}, x::Vector{T})

Predicts the most anomalous factor based on the ranks of the scores.
"""
function predict_ranked(tr_scores::AbstractArray{T,2}, x::Vector{T}) where T
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

"""
	get_subdata(ac, af, mf_scores, mf_labels)
"""
function get_subdata(ac, af, mf_scores, mf_labels)
	true_inds = map(i->mf_labels[i,:] .== ac, 1:3)
    afs = [1,2,3]
    @assert(af in afs)
    nafs = afs[afs.!=af]
    inds = .!true_inds[af] .& true_inds[nafs[1]] .& true_inds[nafs[2]]
    sublabels = mf_labels[:,inds]
    subscores = mf_scores[:,inds]
    subscores, sublabels, inds
end
	
"""
	get_prediction_ranked(ac, af, mf_scores, mf_labels, normal_scores)
"""
function get_prediction_ranked(ac, af, mf_scores, mf_labels, normal_scores)
	subscores, sublabels, inds = get_subdata(ac, af, mf_scores, mf_labels)
	n = size(subscores, 2)
	y_true = ones(Int, n)*af
	y_pred = mapslices(x->predict_anomaly_factor(normal_scores,x), subscores, dims=1)
	acc = mean(y_true .== y_pred)
	return y_true, y_pred, acc
end

function ranked_prediction(lf, model_id, outdir, ac, dataset)
	outf = joinpath(outdir, lf)
	ldata = load(joinpath(ldir, lf))
	mf_scores = ldata[:mf_scores]
	mf_labels = ldata[:mf_labels]

	# first a small experiment - anomalies are in the shape
	normal_scores = hcat(
		ldata[:val_scores][:,ldata[:val_labels] .== 0], 
		ldata[:tst_scores][:,ldata[:tst_labels] .== 0])

	# do the ranked experiment
	results = map(af->get_prediction_ranked(ac, af, mf_scores, mf_labels, normal_scores), 1:3)

	outdf = Dict(
		:model_id => model_id,
		:params => split(lf, ".")[1],
		:y_true_shape => results[1][1],
		:y_true_background => results[2][1],
		:y_true_foreground => results[3][1],
		:y_pred_shape => results[1][2],
		:y_pred_background => results[2][2],
		:y_pred_foreground => results[3][2],
		:acc_shape => results[1][3],
		:acc_background => results[2][3],
		:acc_foreground => results[3][3],
		:mean_acc => mean([x[3] for x in results]),
		:method => "ranked",
		:dataset => dataset,
		:anomaly_class => ac
		)

	save(outf, :df => outdf)
	@info "saved $outf"
	outdf
end

