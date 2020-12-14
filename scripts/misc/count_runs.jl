# this script is intended to provide an overview of number of trained models, evaluated
# scores and the fit and training times
# it is a huge mess, was done at last moment
using DrWatson
@quickactivate
using PrettyTables
using Statistics
using FileIO, BSON, DataFrames
using GenerativeAD
using CSV

# savepath
savepath = datadir("evaluation/experiment_count")
save_df(s,df) = CSV.write(joinpath(savepath, s), df)
load_df(s) = CSV.read(joinpath(savepath, s))


# just count models, score files and training time
# number of models
data_folders = [datadir("experiments/tabular"), datadir("experiments/images"), 
	datadir("experiments/images_leave-one-in")]
#allfiles = vcat(map(f->GenerativeAD.Evaluation.collect_files(f), data_folders)...)

# tabular data
df = load(datadir("evaluation/tabular_eval.bson"))[:df];
ddir = data_folders[1]
models = unique(df.modelname)
models = models[models .!= "vae+ocsvm"]
datasets = unique(df.dataset)

mcount_df_tabular = DataFrame(
	:dataset => datasets)
for model in models
	mcount_df_tabular[Symbol(model)] = zeros(Float32,length(datasets))
end
scount_df_tabular = DataFrame(
	:dataset => datasets)
for model in models
	scount_df_tabular[Symbol(model)] = zeros(Float32,length(datasets))
end

for (di,dataset) in enumerate(datasets)
	for (mi,model) in enumerate(models)
		datapath = joinpath(ddir, model, dataset)
		nseeds = length(readdir(datapath))
		files = GenerativeAD.Evaluation.collect_files(datapath)
		sfiles = filter(x->!occursin("model", x), files)
		mfiles = filter(x->occursin("model", x), files)
		mcount_df_tabular[di,mi+1] = length(mfiles)/nseeds
		scount_df_tabular[di,mi+1] = length(sfiles)/nseeds
	end
end

save_df("model_count_df_tabular_raw.csv", mcount_df_tabular)
save_df("score_count_df_tabular_raw.csv", scount_df_tabular)

mcount_df_tabular = load_df("model_count_df_tabular_raw.csv")
scount_df_tabular = load_df("score_count_df_tabular_raw.csv")

# some models dont save modelfiles
nosave_models = ["MO_GAAL", "abod", "hbos", "if", "knn", "loda", "lof", "ocsvm", 	
	"ocsvm_nu", "ocsvm_rbf", "pidforest"]

for model in nosave_models
	mcount_df_tabular[!,Symbol(model)] .= scount_df_tabular[!,Symbol(model)]
end
mcount_df_tabular.knn .= mcount_df_tabular.knn/3
mcount_df_tabular.vae_knn .= mcount_df_tabular.vae_knn/3

# some dont save score files
nosave_scores = ["vae_knn", "vae_ocsvm"]
for model in nosave_scores
	scount_df_tabular[!,Symbol(model)] .= mcount_df_tabular[!,Symbol(model)]
end
scount_df_tabular.vae_knn .= scount_df_tabular.vae_knn*3

# now save the fit and predict times
fit_ts = []
pred_ts = []
for model in models
	subdf = filter(r->r.modelname==model, df)
	push!(fit_ts, sum(subdf.fit_t) + sum(subdf.fs_fit_t))
	push!(pred_ts, sum(subdf.tr_eval_t) + sum(subdf.tst_eval_t) + sum(subdf.val_eval_t) 
		+ sum(subdf.fs_eval_t))
end
push!(mcount_df_tabular, vcat(["fit_t"], fit_ts))
push!(mcount_df_tabular, vcat(["pred_t"], pred_ts))
#CSV.write("model_count_df_tabular.csv", mcount_df_tabular)
push!(scount_df_tabular, vcat(["fit_t"], fit_ts))
push!(scount_df_tabular, vcat(["pred_t"], pred_ts))
#CSV.write("score_count_df_tabular.csv", scount_df_tabular)
mcount_df_tabular.pidf[1:end-2] .= mcount_df_tabular.pidf[1:end-2]/3

mcount_df_tabular = load_df("model_count_df_tabular.csv")
scount_df_tabular = load_df("score_count_df_tabular.csv")

# rename datasets
dataset_alias = GenerativeAD.Evaluation.DATASET_ALIAS
model_alias = GenerativeAD.Evaluation.MODEL_ALIAS
mcount_df_tabular.dataset .= map(x->get(dataset_alias, x, x), mcount_df_tabular.dataset)
scount_df_tabular.dataset .= map(x->get(dataset_alias, x, x), scount_df_tabular.dataset)

# rename columns
sum_models = Dict(
	"aae" => ["aae", "aae_full", "aae_vamp"],
	"ocsvm" => ["ocsvm", "ocsvm_nu", "ocsvm_rbf"],
	"vae" => ["vae", "vae_full", "vae_simple"],
	"wae" => ["wae", "wae_full", "wae_vamp"]
	)
models_ordered = ["aae", "avae", "gano", "vae", "wae", "abod", "hbos", "if", 
	"knn", "loda", "lof", "orbf", "osvm", "pidf", "maf", "rnvp", "sptn", "fmgn", 
	"gan", "mgal", "dsvd", "vaek", "vaeo"]
mc_final_tabular = DataFrame(:dataset => copy(mcount_df_tabular.dataset))
for model in models_ordered
	ind = findfirst(values(model_alias) .== model)
	mc_final_tabular[!,Symbol(model)] .= 
		mcount_df_tabular[!,Symbol(collect(keys(model_alias))[ind])]
	if model in keys(sum_models)
		for m in sum_models[model][2:end]
			mc_final_tabular[!,Symbol(model)] .+= mcount_df_tabular[!,Symbol(m)]
		end
	end
end
sc_final_tabular = DataFrame(:dataset => copy(scount_df_tabular.dataset))
for model in models_ordered
	ind = findfirst(values(model_alias) .== model)
	sc_final_tabular[!,Symbol(model)] .= 
		scount_df_tabular[!,Symbol(collect(keys(model_alias))[ind])]
	if model in keys(sum_models)
		for m in sum_models[model][2:end]
			sc_final_tabular[!,Symbol(model)] .+= scount_df_tabular[!,Symbol(m)]
		end
	end
end

save_df("model_count_df_tabular.csv", mc_final_tabular)
save_df("score_count_df_tabular.csv", sc_final_tabular)

mc_final_tabular = load_df("model_count_df_tabular.csv")
sc_final_tabular = load_df("score_count_df_tabular.csv")

# now finalize them
mc_final_tabular[1:end-2, 2:end] .= round.(mc_final_tabular[1:end-2, 2:end])
smc_tabular = pretty_table(mc_final_tabular, backend = :latex)
vals=Array(mc_final_tabular[end-1,2:end])/3600
@printf("%.1e &", vals[1])
for val in vals 
	@printf("%.1e & ", val)
end
for val in Array(mc_final_tabular[end,2:end])/3600
	@printf("%.1e & ", val)
end

sc_final_tabular[1:end-2, 2:end] .= round.(sc_final_tabular[1:end-2, 2:end])
ssc_tabular = pretty_table(sc_final_tabular, backend = :latex)

#################
# now the same for image models
df = load(datadir("evaluation/images_eval.bson"))[:df];
ddir = data_folders[2]
models = unique(df.modelname)
datasets = unique(df.dataset)

mcount_df_images = DataFrame(
	:dataset => datasets)
for model in models
	mcount_df_images[Symbol(model)] = zeros(Float32,length(datasets))
end
scount_df_images = DataFrame(
	:dataset => datasets)
for model in models
	scount_df_images[Symbol(model)] = zeros(Float32,length(datasets))
end

for (di,dataset) in enumerate(datasets)
	for (mi,model) in enumerate(models)
		datapath = joinpath(ddir, model, dataset)
		nac = length(readdir(datapath))
		files = GenerativeAD.Evaluation.collect_files(datapath)
		files = filter(x->occursin("seed=1", x), files)
		sfiles = filter(x->!occursin("model", x), files)
		mfiles = filter(x->occursin("model", x), files)
		mcount_df_images[di,mi+1] = length(mfiles)/nac
		scount_df_images[di,mi+1] = length(sfiles)/nac
	end
end

mcount_df_images.knn .= scount_df_images.knn/3
mcount_df_images.ocsvm .= scount_df_images.ocsvm

nosave_scores = ["vae_knn", "vae_ocsvm", "aae_ocsvm"]
for model in nosave_scores
	scount_df_images[!,Symbol(model)] .= mcount_df_images[!,Symbol(model)]
end
scount_df_tabular.vae_knn .= scount_df_tabular.vae_knn*3

save_df("model_count_df_images_raw.csv", mcount_df_images)
save_df("score_count_df_images_raw.csv", scount_df_images)

mcount_df_images = load_df("model_count_df_images_raw.csv")
scount_df_images = load_df("score_count_df_images_raw.csv")


models_ordered = ["aae", "gano", "skip", "vae", "wae", "knn", "osvm", "fano", 
	"fmgn", "dsvd", "vaek", "vaeo"]
mc_final_images = DataFrame(:dataset => copy(mcount_df_images.dataset))
for model in models_ordered
	ind = findlast(values(model_alias) .== model)
	mc_final_images[!,Symbol(model)] .= 
		mcount_df_images[!,Symbol(collect(keys(model_alias))[ind])]
end
sc_final_images = DataFrame(:dataset => copy(scount_df_images.dataset))
for model in models_ordered
	ind = findlast(values(model_alias) .== model)
	sc_final_images[!,Symbol(model)] .= 
		scount_df_images[!,Symbol(collect(keys(model_alias))[ind])]
end

save_df("model_count_df_images.csv", mc_final_images)
save_df("score_count_df_images.csv", sc_final_images)

mc_final_images[1:end, 2:end] .= round.(mc_final_images[1:end, 2:end])
smc_images = pretty_table(mc_final_images, backend = :latex, nosubheader = true)

sc_final_images[1:end, 2:end] .= round.(sc_final_images[1:end, 2:end])
ssc_images = pretty_table(sc_final_images, backend = :latex, nosubheader = true)

fit_t_tabular = sum(df.fit_t)/3600
pred_t_tabular = (sum(df.tr_eval_t) + sum(df.tst_eval_t) + sum(df.val_eval_t))/3600
nexp_images = round(Int,sum(Array(load_df("model_count_df_images.csv")[1:end,2:end]))*10)
nscore_images = round(Int,sum(Array(load_df("score_count_df_images.csv")[1:end,2:end]))*10)


######## LOI ##############
df = load(datadir("evaluation/images_leave-one-in_eval.bson"))[:df];
ddir = data_folders[3]
models = unique(df.modelname)
datasets = unique(df.dataset)

mcount_df_images = DataFrame(
	:dataset => datasets)
for model in models
	mcount_df_images[Symbol(model)] = zeros(Float32,length(datasets))
end
scount_df_images = DataFrame(
	:dataset => datasets)
for model in models
	scount_df_images[Symbol(model)] = zeros(Float32,length(datasets))
end

for (di,dataset) in enumerate(datasets)
	for (mi,model) in enumerate(models)
		datapath = joinpath(ddir, model, dataset)
		nac = length(readdir(datapath))
		files = GenerativeAD.Evaluation.collect_files(datapath)
		files = filter(x->occursin("seed=1", x), files)
		sfiles = filter(x->!occursin("model", x), files)
		mfiles = filter(x->occursin("model", x), files)
		mcount_df_images[di,mi+1] = length(mfiles)/nac
		scount_df_images[di,mi+1] = length(sfiles)/nac
	end
end

mcount_df_images.knn .= scount_df_images.knn/3
mcount_df_images.ocsvm .= scount_df_images.ocsvm

nosave_scores = ["vae_knn", "vae_ocsvm", "aae_ocsvm"]
for model in nosave_scores
	scount_df_images[!,Symbol(model)] .= mcount_df_images[!,Symbol(model)]
end
scount_df_tabular.vae_knn .= scount_df_tabular.vae_knn*3

save_df("model_count_df_images_loi_raw.csv", mcount_df_images)
save_df("score_count_df_images_loi_raw.csv", scount_df_images)

mcount_df_images = load_df("model_count_df_images_loi_raw.csv")
scount_df_images = load_df("score_count_df_images_loi_raw.csv")


models_ordered = ["aae", "gano", "skip", "vae", "wae", "knn", "osvm", "fano", 
	"fmgn", "dsvd", "vaek", "vaeo"]
mc_final_images = DataFrame(:dataset => copy(mcount_df_images.dataset))
for model in models_ordered
	ind = findlast(values(model_alias) .== model)
	mc_final_images[!,Symbol(model)] .= 
		mcount_df_images[!,Symbol(collect(keys(model_alias))[ind])]
end
sc_final_images = DataFrame(:dataset => copy(scount_df_images.dataset))
for model in models_ordered
	ind = findlast(values(model_alias) .== model)
	sc_final_images[!,Symbol(model)] .= 
		scount_df_images[!,Symbol(collect(keys(model_alias))[ind])]
end

save_df("model_count_df_images_loi.csv", mc_final_images)
save_df("score_count_df_images_loi.csv", sc_final_images)

mc_final_images[1:end, 2:end] .= round.(mc_final_images[1:end, 2:end])
smc_images = pretty_table(mc_final_images, backend = :latex, nosubheader = true)

sc_final_images[1:end, 2:end] .= round.(sc_final_images[1:end, 2:end])
ssc_images = pretty_table(sc_final_images, backend = :latex, nosubheader = true)

fit_t_tabular = sum(df.fit_t)/3600
pred_t_tabular = (sum(df.tr_eval_t) + sum(df.tst_eval_t) + sum(df.val_eval_t))/3600
nexp_images_loi = round(Int,sum(Array(load_df("model_count_df_images_loi.csv")[1:end,2:end]))*10)
nscore_images_loi = round(Int,sum(Array(load_df("score_count_df_images_loi.csv")[1:end,2:end]))*10)


# total times and counts
# tabular
df = load(datadir("evaluation/tabular_eval.bson"))[:df];
nexp_tabular = round(Int,sum(Array(load_df("model_count_df_tabular.csv")[1:end-2,2:end]))*5)
nscore_tabular = round(Int,sum(Array(load_df("score_count_df_tabular.csv")[1:end-2,2:end]))*5)

cdf = load_df("model_count_df_tabular.csv")
sdf = load_df("score_count_df_tabular.csv")
multi_models = ["aae", "wae", "vae", "knn", "pidf"]
scales = []
for m in multi_models
	push!(scales, sum(sdf[1:end-2,Symbol(m)])/sum(cdf[1:end-2,Symbol(m)]))
end
fit_t_tabular = 0
for model in unique(df.modelname)
	ft = sum(filter(r->r.modelname==model, df)[!,:fit_t])
	i = findfirst(map(x->occursin(x,model),multi_models))
	if i != nothing && !(occursin("ocsvm", model))
		global fit_t_tabular += ft/scales[i]
	else
		global fit_t_tabular += ft
	end
end
fit_t_tabular = fit_t_tabular/3600/24
pred_t_tabular = (sum(df.tr_eval_t) + sum(df.tst_eval_t) + sum(df.val_eval_t))/3600/24

# images
df = load(datadir("evaluation/images_eval.bson"))[:df];
nexp_images = round(Int,sum(Array(load_df("model_count_df_images.csv")[1:end,2:end]))*10)
nscore_images = round(Int,sum(Array(load_df("score_count_df_images.csv")[1:end,2:end]))*10)

cdf = load_df("model_count_df_images.csv")
sdf = load_df("score_count_df_images.csv")
multi_models = ["aae", "gano", "skip", "wae", "vae", "knn"]
scales = []
for m in multi_models
	push!(scales, sum(sdf[1:end-2,Symbol(m)])/sum(cdf[1:end-2,Symbol(m)]))
end
multi_models2 = ["aae", "GANo", "skip", "wae", "vae", "knn"]
fit_t_images = 0
for model in unique(df.modelname)
	ft = sum(filter(r->r.modelname==model, df)[!,:fit_t])
	i = findfirst(map(x->occursin(x,model),multi_models2))
	if i != nothing && !(occursin("vae_knn", model))
		global fit_t_images += ft/scales[i]
	else
		global fit_t_images += ft
	end
end
fit_t_images = fit_t_images/3600/24
pred_t_images = (sum(df.tr_eval_t) + sum(df.tst_eval_t) + sum(df.val_eval_t))/3600/24

#images LOI
df = load(datadir("evaluation/images_leave-one-in_eval.bson"))[:df];
nexp_images_loi = round(Int,sum(Array(load_df("model_count_df_images_loi.csv")[1:end,2:end]))*10)
nscore_images_loi = round(Int,sum(Array(load_df("score_count_df_images_loi.csv")[1:end,2:end]))*10)

cdf = load_df("model_count_df_images_loi.csv")
sdf = load_df("score_count_df_images_loi.csv")
multi_models = ["aae", "gano", "skip", "wae", "vae", "knn"]
scales = []
for m in multi_models
	push!(scales, sum(sdf[1:end-2,Symbol(m)])/sum(cdf[1:end-2,Symbol(m)]))
end
multi_models2 = ["aae", "GANo", "skip", "wae", "vae", "knn"]
fit_t_images_loi = 0
for model in unique(df.modelname)
	ft = sum(filter(r->r.modelname==model, df)[!,:fit_t])
	i = findfirst(map(x->occursin(x,model),multi_models2))
	if i != nothing && !(occursin("vae_knn", model))
		global fit_t_images_loi += ft/scales[i]
	else
		global fit_t_images_loi += ft
	end
end
fit_t_images_loi = fit_t_images_loi/3600/24
pred_t_images_loi = (sum(df.tr_eval_t) + sum(df.tst_eval_t) + sum(df.val_eval_t))/3600/24


total_nmodels = nexp_images + nexp_images_loi + nexp_tabular
total_nscores = nscore_images + nscore_images_loi + nscore_tabular
total_fit_t = fit_t_tabular + fit_t_images + fit_t_images_loi
total_pred_t = pred_t_tabular + pred_t_images + pred_t_images_loi
