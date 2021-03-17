"""
	experiment(score_fun, parameters, data, savepath; save_entries...)
Eval score function on test/val/train data and save.
"""
function experiment(score_fun, parameters, data, savepath; verb=true, save_result=true, save_entries...)
	tr_data, val_data, tst_data = data

	# extract scores
	tr_scores, tr_eval_t, _, _, _ = @timed score_fun(tr_data[1])
	val_scores, val_eval_t, _, _, _ = @timed score_fun(val_data[1])
	tst_scores, tst_eval_t, _, _, _ = @timed score_fun(tst_data[1])

	# now save the stuff
	savef = joinpath(savepath, savename(parameters, "bson", digits=5))
	result = (
		parameters = parameters,
		tr_scores = tr_scores,
		tr_labels = tr_data[2], 
		tr_eval_t = tr_eval_t,
		val_scores = val_scores,
		val_labels = val_data[2], 
		val_eval_t = val_eval_t,
		tst_scores = tst_scores,
		tst_labels = tst_data[2], 
		tst_eval_t = tst_eval_t
		)
	result = Dict{Symbol, Any}([sym=>val for (sym,val) in pairs(merge(result, save_entries))]) # this has to be a Dict 
	if save_result
		tagsave(savef, result, safe = true)
		verb ? (@info "Results saved to $savef") : nothing
	end
	result
end

"""
	edit_params(data, parameters)
This modifies parameters according to data. Default version only returns the input arg. 
Overload for models where this is needed.
"""
function edit_params(data, parameters)
	parameters
end

"""
	sample_params(parameters_range; add_model_seed=false)
Samples a named tuple from a given parameters_range tuple. If a model has the option
of a fixed initial seed, set `add_model_seed` to true to add random integer entry `init_seed`.
"""
function sample_params(parameters_range; add_model_seed=false)
	p = (;zip(keys(parameters_range), map(x->sample(x, 1)[1], parameters_range))...)
	add_model_seed ? merge((;init_seed=rand(1:Int(1e8))), p) : p
end


"""
	check_params(savepath, parameters)

Returns `true` if the model with given parameters wasn't already trained and saved. 
"""
function check_params(savepath, parameters)
	if ~isdir(savepath)
		return true
	end
	# filter out duplicates created by tagsave
	fs = filter(x->!(occursin("_#", x)), readdir(savepath))
	# filter out model files
	fs = filter(x->!(startswith(x, "model")), fs)
	# if the first argument name contains a "_", than the savename is parsed wrongly
	saved_params = map(x -> DrWatson.parse_savename("_"*x)[2], fs)
	# now filter out saved models where parameter names are different or missing
	pkeys = collect(keys(parameters))
	filter!(ps->intersect(pkeys, Symbol.(collect(keys(ps))))==pkeys, saved_params)
	for params in saved_params
		all(map(k->params[String(k)] == parameters[k], pkeys)) ? (return false) : nothing
	end
	true
end