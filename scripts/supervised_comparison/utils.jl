function basic_stats(labels, scores)
	try
		roc = EvalMetrics.roccurve(labels, scores)
		auc = EvalMetrics.auc_trapezoidal(roc...)
		prc = EvalMetrics.prcurve(labels, scores)
		auprc = EvalMetrics.auc_trapezoidal(prc...)

		t5 = EvalMetrics.threshold_at_fpr(labels, scores, 0.05)
		cm5 = ConfusionMatrix(labels, scores, t5)
		tpr5 = EvalMetrics.true_positive_rate(cm5)
		f5 = EvalMetrics.f1_score(cm5)

		return auc, auprc, tpr5, f5
	catch e
		if isa(e, ArgumentError)
			return NaN, NaN, NaN, NaN
		else
			rethrow(e)
		end
	end
end

auc_val(labels, scores) = EvalMetrics.auc_trapezoidal(EvalMetrics.roccurve(labels, scores)...)

function perf_at_p_new(p, p_normal, val_scores, val_y, tst_scores, tst_y, init_alpha, base_beta; 
	seed=nothing, scale=true, kwargs...)
	scores, labels, _ = try
		_subsample_data(p, p_normal, val_y, val_scores; seed=seed)
	catch e
		return NaN, NaN
	end
	# if there are no positive samples return NaNs
	if sum(labels) == 0
		val_auc = NaN
		tst_auc = auc_val(tst_y, tst_scores[:,1])
	# if top samples are only positive
	# we cannot train alphas
	# therefore we return the default val performance
	elseif sum(labels) == length(labels) 
		val_auc = NaN
		tst_auc = auc_val(tst_y, tst_scores[:,1])
	# if they are not only positive, then we train alphas and use them to compute 
	# new scores - auc vals on the partial validation and full test dataset
	else
		try
			# get the logistic regression model
            model = if method == "logreg"
                LogReg()
            elseif method == "probreg"
                ProbReg()
            elseif method == "robreg"
                RobReg(alpha=init_alpha, beta=base_beta/sum(labels))
            else
                error("unknown method $method")
            end

            # fit
            if method == "logreg"
                fit!(model, scores, labels)
            elseif method == "probreg"
                fit!(model, scores, labels; verb=false, early_stopping=true, patience=10, balanced=true)
            elseif method == "robreg"
            	try
            		fit!(model, scores, labels; verb=false, early_stopping=true, scale=scale, patience=10,
                    balanced=true)
                catch e
	            	if isa(e, PyCall.PyError)
			            return NaN, NaN
			        else
			        	rethrow(e)
			        end
			    end 
            end

            # predict
			val_probs = predict(model, scores, scale=scale)
			tst_probs = predict(model, tst_scores, scale=scale)
			val_auc = auc_val(labels, val_probs)
			tst_auc = auc_val(tst_y, tst_probs)
		catch e
			if isa(e, LoadError) || isa(e, ArgumentError)
				val_prec = NaN
				val_auc = NaN
				tst_auc = auc_val(tst_y, tst_scores[:,1])
			else
				rethrow(e)
			end
		end
	end
	return val_auc, tst_auc
end	

nanmean(x) = mean(x[.!isnan.(x)])

function perf_at_p_agg(args...; kwargs...)
	results = [perf_at_p_new(args...;seed=seed, kwargs...) for seed in 1:max_seed_perf]
	return nanmean([x[1] for x in results]), nanmean([x[2] for x in results])
end

# get the right lf when using a selection of best models
function get_latent_file(_params, lfs)
	if _params["latent_score_type"] != latent_score_type
		return nothing
	end

	model_id = _params["init_seed"]
	_lfs = filter(x->occursin("$(model_id)",x), lfs)
	_lfs = if _params["latent_score_type"] == "knn"
		k = _params["k"]
		v = _params["v"]
		filter(x->occursin("k=$(k)_",x) && occursin("v=$v",x), _lfs)
	else
		_lfs
	end
	if length(_lfs) != 1
		error("something wrong when processing $(_params)")
	end
	return _lfs[1]
end

function prepare_savefile(save_dir, lf, base_beta, method)
	outf = joinpath(save_dir, split(lf, ".")[1])
	outf *= "_beta=$(base_beta)_method=$(method).bson"
	@info "Working on $outf"
	if !force && isfile(outf)
		@info "Already present, skipping."
        return ""
	end	
	return outf
end

function load_scores(model_id, lf, latent_dir, rfs, res_dir)
	# load the saved scores
	ldata = load(joinpath(latent_dir, lf))
	rf = filter(x->occursin("$(model_id)", x), rfs)
	if length(rf) < 1
		@info "Something is wrong, original score file for $lf not found"
		return
	end
	rf = rf[1]
	rdata = load(joinpath(res_dir, rf))

	# prepare the data
	if isnan(ldata[:val_scores][1])
		@info "Score data not found or corrupted"
		return
	end
	if isnothing(rdata[:val_scores]) || isnothing(rdata[:tst_scores])
		@info "Normal score data not found"
		return
	end

	scores_val = cat(rdata[:val_scores], transpose(ldata[:val_scores]), dims=2);
	scores_tst = cat(rdata[:tst_scores], transpose(ldata[:tst_scores]), dims=2);
	y_val = ldata[:val_labels];
	y_tst = ldata[:tst_labels];

	return scores_val, scores_tst, y_val, y_tst, ldata, rdata
end

function original_class_split(dataset, ac; seed=1, ratios=(0.6,0.2,0.2))
	# get the original data with class labels
	if dataset == "wildlife_MNIST"
		(xn, yn), (xa, ya) = GenerativeAD.Datasets.load_wildlife_mnist_data(normal_class_ind=ac);
	else
		throw("Dataset $dataset not implemented")
	end
	# then get the original labels in the same splits as we have the scores
	(c_tr, y_tr), (c_val, y_val), (c_tst, y_tst) = GenerativeAD.Datasets.train_val_test_split(yn,ya,ratios; seed=seed)
	return (c_tr, y_tr), (c_val, y_val), (c_tst, y_tst)
end

function basic_experiment(val_scores, val_y, tst_scores, tst_y, outf, base_beta, init_alpha, 
	scale, dataset, rdata, ldata, seed, ac, method, score_type, latent_score_type)
	# setup params
	parameters = merge(ldata[:parameters], (beta=base_beta, init_alpha=init_alpha, scale=scale))
	save_modelname = modelname*"_$method"

	res_df = @suppress begin
		# prepare the result dataframe
		res_df = OrderedDict()
		res_df["modelname"] = save_modelname
		res_df["dataset"] = dataset
		res_df["phash"] = GenerativeAD.Evaluation.hash(parameters)
		res_df["parameters"] = "_"*savename(parameters)
		res_df["fit_t"] = rdata[:fit_t]
		res_df["tr_eval_t"] = ldata[:tr_eval_t] + rdata[:tr_eval_t]
		res_df["val_eval_t"] = ldata[:val_eval_t] + rdata[:val_eval_t]
		res_df["tst_eval_t"] = ldata[:tst_eval_t] + rdata[:tst_eval_t]
		res_df["seed"] = seed
		res_df["npars"] = rdata[:npars]
		res_df["anomaly_class"] = ac
		res_df["method"] = method
		res_df["score_type"] = score_type
		res_df["latent_score_type"] = latent_score_type

		# fit the logistic regression - first on all the validation data
		# first, filter out NaNs and Infs
		inds = vec(mapslices(r->!any(r.==Inf), val_scores, dims=2))
		val_scores = val_scores[inds, :]
		val_y = val_y[inds]
		inds = vec(mapslices(r->!any(isnan.(r)), val_scores, dims=2))
		val_scores = val_scores[inds, :]
		val_y = val_y[inds]

        # get the logistic regression model - scale beta by the number of anomalies
        model = RobReg(alpha=init_alpha, beta=base_beta/sum(val_y))
        
        # fit
        converged = true
    	try
            fit!(model, val_scores, val_y; verb=false, early_stopping=true, scale=scale, patience=10,
                balanced=true)
        catch e
        	if isa(e, PyCall.PyError)
        		converged = false
	        else
	        	rethrow(e)
	        end
	    end
        if converged
	        val_probs = predict(model, val_scores, scale=scale)
	        tst_probs = predict(model, tst_scores, scale=scale)
			
			# now fill in the values
			res_df["val_auc"], res_df["val_auprc"], res_df["val_tpr_5"], res_df["val_f1_5"] = 
				basic_stats(val_y, val_probs)
			res_df["tst_auc"], res_df["tst_auprc"], res_df["tst_tpr_5"], res_df["tst_f1_5"] = 
				basic_stats(tst_y, tst_probs)
		else
			res_df["val_auc"], res_df["val_auprc"], res_df["val_tpr_5"], res_df["val_f1_5"] = 
				NaN, NaN, NaN, NaN
			res_df["tst_auc"], res_df["tst_auprc"], res_df["tst_tpr_5"], res_df["tst_f1_5"] = 
				NaN, NaN, NaN, NaN
		end

		# then do the same on a small section of the data
		ps = [100.0, 50.0, 20.0, 10.0, 5.0, 2.0, 1.0, 0.5, 0.2, 0.1]
		auc_ano_100 = [perf_at_p_agg(p/100, 1.0, val_scores, val_y, tst_scores, tst_y, init_alpha, 
            base_beta; scale=scale) for p in ps]
		for (k,v) in zip(map(x->x * "_100", AUC_METRICS), auc_ano_100)
			res_df["val_"*k] = v[1]
			res_df["tst_"*k] = v[2]
		end

		res_df
	end
	
	# then save it
	res_df = DataFrame(res_df)
	save(outf, Dict(:df => res_df))
	#@info "Saved $outf."
	res_df
end
