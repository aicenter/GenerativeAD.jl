"""
	experiment(score_fun, parameters, data, savepath; save_entries...)

Eval score function on test/val/train data and save.
"""
function experiment(score_fun, parameters, data, savepath; verb=true, save_entries...)
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
	tagsave(savef, result, safe = true)
	verb ? (@info "Results saved to $savef") : nothing
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
	check_params(edit_params_f, savepath, data, parameters)

This checks if the model with given parameters wasn't already trained and saved. 
"""
function check_params(savepath, data, parameters)
	if ~isdir(savepath)
		return true
	end
	# filter out duplicates created by tagsave
	fs = filter(x->!(occursin("_#", x)), readdir(savepath))
	# filter out model files
	fs = filter(x->!(startswith(x, "model")), fs)
	# if the first argument name contains a "_", than the savename is parsed wrongly
	saved_params = map(x -> DrWatson.parse_savename("_"*x)[2], fs)
	for params in saved_params
		all(map(k->params[String(k)] == parameters[k], collect(keys(parameters)))) ? (return false) : nothing
	end
	true
end
