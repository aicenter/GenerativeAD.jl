"""
	experiment(score_fun, parameters, data, savepath; save_entries...)

Eval score function on test/val/train data and save.
"""
function experiment(score_fun, parameters, data, savepath; save_entries...)
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
	result
end
