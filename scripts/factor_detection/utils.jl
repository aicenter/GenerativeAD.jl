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
	percentiles = maxinds/n
	y_pred = argmax(maxinds)
	y_pred, percentiles
end
function predict_ranked(tr_scores::AbstractArray{T,2}, tst_scores::AbstractArray{T,2}) where T
	res = mapslices(x->predict_ranked(tr_scores,x), tst_scores, dims=1)
	return vec([x[1] for x in res]), hcat([x[2] for x in res]...)
end


"""
	predict_random(x::Vector)

Prediction of a random class.	
"""
predict_random(x::Vector) = sample(1:length(x))

function get_subinds(ac, af, mf_labels)
	true_inds = map(i->mf_labels[i,:] .== ac, 1:3)
    afs = [1,2,3]
    @assert(af in afs)
    nafs = afs[afs.!=af]
    inds = .!true_inds[af] .& true_inds[nafs[1]] .& true_inds[nafs[2]]
end

"""
	get_subdata(ac, af, mf_scores, mf_labels)
"""
function get_subdata(ac, af, mf_scores, mf_labels)
	inds = get_subinds(ac, af, mf_labels)
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
	y_pred, percentiles = predict_ranked(normal_scores, subscores)
	acc = mean(y_true .== y_pred)
	return y_true, y_pred, percentiles, acc
end

function ranked_prediction(lf, model_id, outdir, ac, dataset)
	outf = joinpath(outdir, lf)
	ldata = load(joinpath(ldir, lf))
	mf_scores = ldata[:mf_scores]
	mf_labels = ldata[:mf_labels]

	# first a small experiment - anomalies are in the shape
	normal_scores = ldata[:val_scores][:,ldata[:val_labels] .== 0]

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
		:percentiles_shape => results[1][3],
		:percentiles_background => results[2][3],
		:percentiles_foreground => results[3][3],
		:acc_shape => results[1][4],
		:acc_background => results[2][4],
		:acc_foreground => results[3][4],
		:mean_acc => mean([x[4] for x in results]),
		:method => "ranked",
		:dataset => dataset,
		:anomaly_class => ac
		)

	save(outf, :df => outdf)
	@info "saved $outf"
	outdf
end

function compute_bfscores(model, x)
	px = jl2py(x)
	rx = py2jl(model.model.reconstruct(px));
	mask, background, foreground = map(py2jl, model.model.sgvae(px));
	bscore = sum(((rx .- x) .* (1 .-  mask)) .^2) / sum(1 .- mask)
	fscore = sum(((rx .- x) .* mask ) .^ 2) / sum(mask)
	return bscore, fscore
end

function predict_masked(model,x::AbstractArray{T,3}; n=10) where T
	x = reshape(x,size(x)...,1)
	bfscores = []
	for i in 1:n
		try
			bfscore = compute_bfscores(model, x)
			push!(bfscores, bfscore)
		catch e 
			nothing
		end
	end
	if length(bfscores) == 0
		return NaN, NaN, NaN
	end
	bscore = mean(vec([y[1] for y in bfscores]))
	fscore = mean(vec([y[2] for y in bfscores]))
	# 2 is for background, 3 is for foreground
	return 1+argmax((bscore, fscore)), bscore, fscore
end
function predict_masked(model,x::AbstractArray{T,4}) where T
	result = map(i->predict_masked(model, x[:,:,:,i]), 1:size(x,4))
	return [x[1] for x in result], [x[2] for x in result], [x[3] for x in result] 
end

function get_prediction_masked(model, ac, af, mf_X, mf_Y)
	subinds = get_subinds(ac, af, mf_Y);
	tst_y = mf_Y[:,subinds]
	tst_x = mf_X[:,:,:,subinds];
	n = size(tst_x, 4)
	y_true = ones(Int, n)*af
	y_pred, bscores, fscores = predict_masked(model, tst_x)
	acc = mean(y_true .== y_pred)
	y_true, y_pred, bscores, fscores, acc	
end
