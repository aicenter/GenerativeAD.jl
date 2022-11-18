function fit_predict_lrnormal(val_scores, tst_scores, val_y, tst_y, seed, p, p_normal)
	_val_scores, _val_y, _ = _subsample_data(p, p_normal, val_y, val_scores; seed=seed)
	_tst_scores, _tst_y, _ = _subsample_data(p, p_normal, tst_y, tst_scores; seed=seed)
	return fit_predict_lrnormal(_val_scores, _tst_scores, _val_y, _tst_y)
end

function fit_predict_lrnormal(val_scores, tst_scores, val_y, tst_y)
	_init_alpha, _alpha0 = compute_alphas(val_scores, val_y)
	lrmodel = RobReg(input_dim = size(_val_scores,2), alpha=_init_alpha, alpha0=_alpha0, 
	                	beta=base_beta/sum(_val_y))
	fit!(lrmodel, _val_scores, _val_y; verb=false, early_stopping=true, scale=scale, patience=10,
		balanced=true)

	# predict
	val_probs = predict(lrmodel, val_scores, scale=scale)
	tst_probs = predict(lrmodel, tst_scores, scale=scale)
	val_auc = auc_val(val_y, val_probs)
	tst_auc = auc_val(tst_y, tst_probs)
	return val_auc, tst_auc, lrmodel.alpha
end

function get_basic_scores(model_id, ac, ps, datatype="leave-one-in", base_modelname="sgvaegan100",
	dataset="SVHN2")
	latent_dir = datadir("sgad_latent_scores/images_$(datatype)/$(base_modelname)/$(dataset)/ac=$(ac)/seed=1")
	lfs = readdir(latent_dir)
	ltypes = map(lf->split(split(lf, "score=")[2], ".")[1], lfs)
	lfs = lfs[ltypes .== latent_score_type]
	lparams = map(x->parse_savename(x)[2], lfs)
	ilf = findfirst([x["v"] == ps["v"] && x["k"] == ps["k"] && x["id"] == ps["init_seed"] for x in lparams])
	lf = lfs[ilf]

	# top score files
	res_dir = datadir("experiments/images_$(datatype)/$(base_modelname)/$(dataset)/ac=$(ac)/seed=1")
	rfs = readdir(res_dir)

	# get the saved scores
	val_scores, tst_scores, val_y, tst_y, ldata, rdata = 
		load_scores(model_id, lf, latent_dir, rfs, res_dir, base_modelname)
	
	inds = vec(mapslices(r->!any(r.==Inf), val_scores, dims=2))
	val_scores = val_scores[inds, :]
	val_y = val_y[inds]
	inds = vec(mapslices(r->!any(isnan.(r)), val_scores, dims=2))
	val_scores = val_scores[inds, :]
	val_y = val_y[inds]

	val_scores, tst_scores, val_y, tst_y	
end
