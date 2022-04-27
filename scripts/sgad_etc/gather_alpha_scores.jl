# this is meant to gather the alpha scores in such a way so that it can be used to generate knowledge plots
using DrWatson
@quickactivate
using GenerativeAD
using BSON, FileIO, DataFrames
using ArgParse
using OrderedCollections

s = ArgParseSettings()
@add_arg_table! s begin
   "modelname"
        default = "sgvae"
        arg_type = String
        help = "modelname"
    "dataset"
        default = "wildlife_MNIST"
        arg_type = String
        help = "dataset or mvtec category"
    "datatype"
        default = "leave-one-in"
        arg_type = String
        help = "leave-one-in or mvtec"
    "anomaly_class"
    	default = nothing
    	help = "which class to compute"
    "--valauc"
    	action = :store_true
    	help = "use validation auc instead of val precision"
end
parsed_args = parse_args(ARGS, s)
@unpack modelname, dataset, datatype, anomaly_class, valauc = parsed_args
max_ac = (datatype == "mvtec") ? 1 : 10
max_seed = (datatype == "mvtec") ? 5 : 1
acs = isnothing(anomaly_class) ? collect(1:max_ac) : [Meta.parse(anomaly_class)]
valauc_suffix = valauc ? "_auc" : ""
# only compute all the subtypes for the precision implementation
subtypes = valauc ? [""] : ["", "_normal", "_kld", "_normal_logpx", "_knn"]

function create_save_scores(model_id, af, out_model_name, alpha_dir, pdata, dataset, seed, ac, save_dir, st)
	adata = load(joinpath(alpha_dir, af))[:df]
	if st != ""
		@assert adata["latent_score_type"][1] == st
	end

	# this is gonna be used at all thresholds
	cres_df = OrderedDict()
	cres_df["modelname"] = out_model_name
	cres_df["dataset"] = dataset == "metal_nut" ? "nut" : dataset 
	cres_df["phash"] = nothing
	cres_df["parameters"] = nothing
	cres_df["fit_t"] = adata["fit_t"][1]
	cres_df["tr_eval_t"] = adata["tr_eval_t"][1]
	cres_df["val_eval_t"] = adata["val_eval_t"][1]
	cres_df["tst_eval_t"] = adata["tst_eval_t"][1]
	cres_df["seed"] = seed
	cres_df["npars"] = adata["npars"][1]
	if datatype != "mvtec"
		cres_df["anomaly_class"] = ac
	end
	cres_df["fs_fit_t"] = 0.0
	cres_df["fs_eval_t"] = 0.0
	for n in names(pdata)[14:end]
		cres_df[n] = NaN
	end
	cres_df["val_auprc"] = adata["val_auprc"][1]
	cres_df["val_tpr_5"] = adata["val_tpr_5"][1]
	cres_df["val_f1_5"] = adata["val_f1_5"][1]
	cres_df["tst_auprc"] = adata["tst_auprc"][1]
	cres_df["tst_tpr_5"] = adata["tst_tpr_5"][1]
	cres_df["tst_f1_5"] = adata["tst_f1_5"][1]

	# now this is going to be different for each
	for p in [0.01, 0.05, 0.1, 0.2, 1.0]
		ip = p >= 0.01 ? 1 : 2
		sp = split("$(p*100)", ".")[ip]
		params = adata["parameters"][1]
		params = params * "_method=$(adata["method"][1])" * "_p=$sp"

		res_df = deepcopy(cres_df)
		res_df["phash"] = GenerativeAD.Evaluation.hash(params)
		res_df["parameters"] = params
		if p == 1.0
			res_df["val_auc"] = adata["val_auc"][1]
			res_df["tst_auc"] = adata["tst_auc"][1]
		else
			if valauc 
				res_df["val_pat_$(sp)"] = adata["val_auc_$(sp)"][1]
			else # this is the original method
				res_df["val_pat_$(sp)"] = adata["val_pat_$(sp)"][1]
			end
			res_df["tst_auc"] = adata["tst_auc_$(sp)"][1]
		end
		res_df = DataFrame(res_df)

		# outf
		outf = "eval_model_id=$(model_id)"
		outf *= split(split(af, ".")[1], "model_id=$(model_id)")[2]
		outf *= "_p=$sp.bson"

		# save it
		save(joinpath(save_dir, outf), Dict(:df=>res_df))
		#@info "Saved $(joinpath(save_dir, outf))."
	end
end

# we need to load an eval file that is actually used in the knowledge plot generation
pf = datadir("sgad_alpha_evaluation/prototype.bson")
pdata = load(pf)[:df]

for ac in acs
	for seed in 1:max_seed
		# first do the one for full validation dataset
		for sub_type in subtypes
			aggreg_type = "alpha"*sub_type
			out_model_name = modelname*"_"*aggreg_type*valauc_suffix

			# now to emulate this
			alpha_dir = datadir("sgad_alpha_evaluation/images_$(datatype)/$(modelname)/$(dataset)/ac=$(ac)/seed=$(seed)")
			afs = readdir(alpha_dir)
			if sub_type != ""
				st = sub_type[2:end]
				afs = filter(af -> split(split(af, "score=")[2], "_method=")[1]==st, afs)
			else
				st = ""
			end
			model_ids = map(x->Meta.parse(split(split(x, "=")[2], "_")[1]), afs)

			# save path
			save_dir = datadir("evaluation/images_$(datatype)/$(modelname)_$(aggreg_type)$(valauc_suffix)/$(dataset)/ac=$(ac)/seed=$(seed)")
			mkpath(save_dir)
			@info "Saving data to $(save_dir)..."

			for (model_id, af) in zip(model_ids, afs)
				create_save_scores(model_id, af, out_model_name, alpha_dir, pdata, dataset, seed, ac, save_dir, st)
			end
			@info "Done."
		end
	end
end
