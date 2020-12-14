# this produces critical diagrams, however the rank values are input by hand, 
# so thos needs to have some more tuning
using DrWatson
@quickactivate
using GenerativeAD

# save path
savepath = datadir("evaluation/cd_diagrams")
mkpath(savepath)

# number of datasets
nd = 40

# tabular
model_names = ["aae", "avae", "gano", "vae", "wae", "abod", "hbos", "if", "knn", 
"loda", "lof", "orbf", "osvm", "pidf", "maf", "rnvp", "sptn", "fmgn", "gan", 
"mgal", "dsvd", "vaek", "vaeo"]
model_names_gray = ["aae", "avae", "gano", "vae", "wae", "\\textcolor{gray}{abod}", 
"\\textcolor{gray}{hbos}", "\\textcolor{gray}{if}", "\\textcolor{gray}{knn}", 
"\\textcolor{gray}{loda}", "\\textcolor{gray}{lof}", "\\textcolor{gray}{orbf}", 
"\\textcolor{gray}{osvm}", "\\textcolor{gray}{pidf}", "maf", "rnvp", "sptn", "fmgn", 
"gan", "mgal", "dsvd", "vaek", "vaeo"]
model_ranks = [8.8, 13.4, 10.2, 8.0, 6.3, 11.3, 14.4, 14.2, 8.0, 15.6, 11.3, 8.1, 2.8, 
	14.1, 8.8, 8.7, 10.3, 11.3, 11.8, 21.4, 16.0, 10.9, 6.7]
plot5 = GenerativeAD.Evaluation.ranks2tikzcd(model_ranks, model_names_gray, 
	GenerativeAD.Evaluation.nemenyi_cd(length(model_ranks), nd, 0.05); 
	scale=0.45, vpad=0.6, levelwidth=0.03, vdist=0.15)
GenerativeAD.Evaluation.string2file(joinpath(savepath, "tabular_auc_auc_cdd5.tex"), plot5)

plot10 = GenerativeAD.Evaluation.ranks2tikzcd(model_ranks, model_names_gray, 
	GenerativeAD.Evaluation.nemenyi_cd(length(model_ranks), nd, 0.1); 
	scale=0.45, vpad=0.6, levelwidth=0.03, vdist=0.15)
GenerativeAD.Evaluation.string2file(joinpath(savepath, "tabular_auc_auc_cdd10.tex"), plot10)

#
model_ranks_tpr5 = [10.5, 12.5, 9.9, 7.6, 8.2, 12.0, 14.7, 15.9, 9.8, 17.9, 10.9, 10.2, 
	3.0, 13.9, 10.0, 9.8, 11.8, 9.6, 10.0, 18.9, 15.9, 11.2, 6.4]
plot5tpr = GenerativeAD.Evaluation.ranks2tikzcd(model_ranks_tpr5, model_names_gray, 
	GenerativeAD.Evaluation.nemenyi_cd(length(model_ranks), nd, 0.05); 
	scale=0.55, vpad=0.5, levelwidth=0.03, vdist=0.13)
GenerativeAD.Evaluation.string2file(joinpath(savepath, "tabular_tpr5_tpr5_cdd5.tex"), 
	plot5tpr)

plot10tpr = GenerativeAD.Evaluation.ranks2tikzcd(model_ranks_tpr5, model_names_gray, 
	GenerativeAD.Evaluation.nemenyi_cd(length(model_ranks), nd, 0.1); 
	scale=0.55, vpad=0.5, levelwidth=0.03, vdist=0.13)
GenerativeAD.Evaluation.string2file(joinpath(savepath, "tabular_tpr5_tpr5_cdd10.tex"), 
	plot10tpr)

# image data
nd = 40
img_model_names = ["aae", "gano", "skip", "vae", "wae", "\\textcolor{gray}{knn}", 
"\\textcolor{gray}{osvm}", "fano", "fmgn", "dsvd", "vaek", "vaeo"]
img_model_ranks = [3.0, 6.2, 11.8, 6.0, 8.2, 8.0, 7.2, 5.8, 3.0, 6.2, 5.0, 2.5]
plot5img = GenerativeAD.Evaluation.ranks2tikzcd(img_model_ranks,img_model_names, 
	GenerativeAD.Evaluation.nemenyi_cd(length(img_model_ranks), nd, 0.05))
GenerativeAD.Evaluation.string2file(joinpath(savepath, "images_auc_auc_cdd5.tex"), 
	plot5img)

plot10img = GenerativeAD.Evaluation.ranks2tikzcd(img_model_ranks,img_model_names, 
	GenerativeAD.Evaluation.nemenyi_cd(length(img_model_ranks), nd, 0.1))
GenerativeAD.Evaluation.string2file(joinpath(savepath, "images_auc_auc_cdd10.tex"), 
	plot10img)

plot10img = GenerativeAD.Evaluation.ranks2tikzcd(img_model_ranks,img_model_names, 
	GenerativeAD.Evaluation.nemenyi_cd(length(img_model_ranks), 4, 0.1))
GenerativeAD.Evaluation.string2file(
	joinpath(savepath, "images_auc_auc_cdd10_4_datasets.tex"), plot10img)


# img data per class
nd = 40
img_model_names = ["aae", "gano", "skip", "vae", "wae", "\\textcolor{gray}{knn}", 
"\\textcolor{gray}{osvm}", "fano", "fmgn", "dsvd", "vaek", "vaeo"]
img_model_ranks_per_class = [5.0, 6.8, 10.6, 5.8, 6.2, 6.3, 7.4, 6.4, 
	4.6, 6.5, 3.9, 3.2]
plot10img = GenerativeAD.Evaluation.ranks2tikzcd(img_model_ranks_per_class,img_model_names, 
	GenerativeAD.Evaluation.nemenyi_cd(length(img_model_ranks_per_class), nd, 0.1))
GenerativeAD.Evaluation.string2file(joinpath(savepath, "images_auc_auc_cdd10_per_class.tex"), 
	plot10img)
