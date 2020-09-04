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
	savef = joinpath(savepath, savename(parameters, "bson"))
	result = Dict(
		:parameters => parameters,
		:tr_scores => tr_scores,
		:tr_labels => tr_data[2], 
		:tr_eval_t => tr_eval_t,
		:val_scores => val_scores,
		:val_labels => val_data[2], 
		:val_eval_t => val_eval_t,
		:tst_scores => tst_scores,
		:tst_labels => tst_data[2], 
		:tst_eval_t => tst_eval_t
		)
	result = merge(result, save_entries)
	tagsave(savef, result, safe = true)
	verb ? (@info "Results saved to $savef") : nothing
	result
end

"""
	check_params(edit_params_f, savepath, data, parameters)

This checks if the model with given parameters wasn't already trained and saved. 
"""
function check_params(edit_params_f, savepath, data, parameters)
	eparams = edit_params_f(data, parameters)
	if ~isdir(savepath)
		return true
	end
	saved_params = map(x -> DrWatson.parse_savename(x)[2], readdir(savepath))
	for params in saved_params
		all(map(k->params[String(k)] == eparams[k], collect(keys(eparams)))) ? (return false) : nothing
	end
	return true
end
