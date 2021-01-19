using Statistics
using Distributions

# R = mean ranks, n = number of datasets, k = number of models
friedman_test_statistic(R::Vector,n::Int,k::Int) = 12*n/(k*(k+1))*(sum(R.^2) - k*(k+1)^2/4)
# k = number of models
crit_chisq(α::Real, df::Int) = quantile(Chisq(df), 1-α)
friedman_critval(α::Real, k::Int) = crit_chisq(α/2, k-1)


# k = number of groups, df = (N - k) = (total samples - k)
crit_srd(α::Real, k::Real, df::Real) = 
	(isnan(k) | isnan(df)) ? NaN : quantile(StudentizedRange(df, k), 1-α)
nemenyi_cd(k::Int, n::Int, α::Real) = sqrt(k*(k+1)/(6*n))*crit_srd(α, k, Inf)/sqrt(2)

# this checks out with our kdd18 paper
nd = 35
nm = 6
α = 0.05
nemenyi_cd(nm, nd, α) # = 1.275

# this is for our new paper
nd = 40
nm = 23
α = 0.05
ncv5 = nemenyi_cd(nm, nd, α) # = 5.48
α = 0.1
ncv10 = nemenyi_cd(nm, nd, α) # = 5.14

include("crit_diag_functions.jl")
nd = 40
#
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
plot5 = ranks2tikzcd(model_ranks, model_names_gray, 
	nemenyi_cd(length(model_ranks), nd, 0.05); 
	scale=0.45, vpad=0.6, levelwidth=0.03, vdist=0.15)
string2file("tabular_auc_auc_cdd5.tex", plot5)

plot10 = ranks2tikzcd(model_ranks, model_names_gray, 
	nemenyi_cd(length(model_ranks), nd, 0.1); 
	scale=0.45, vpad=0.6, levelwidth=0.03, vdist=0.15)
string2file("tabular_auc_auc_cdd10.tex", plot10)

#
model_ranks_tpr5 = [10.5, 12.5, 9.9, 7.6, 8.2, 12.0, 14.7, 15.9, 9.8, 17.9, 10.9, 10.2, 
	3.0, 13.9, 10.0, 9.8, 11.8, 9.6, 10.0, 18.9, 15.9, 11.2, 6.4]
plot5tpr = ranks2tikzcd(model_ranks_tpr5, model_names_gray, 
	nemenyi_cd(length(model_ranks), nd, 0.05); 
	scale=0.55, vpad=0.5, levelwidth=0.03, vdist=0.13)
string2file("tabular_tpr5_tpr5_cdd5.tex", plot5tpr)

plot10tpr = ranks2tikzcd(model_ranks_tpr5, model_names_gray, 
	nemenyi_cd(length(model_ranks), nd, 0.1); 
	scale=0.55, vpad=0.5, levelwidth=0.03, vdist=0.13)
string2file("tabular_tpr5_tpr5_cdd10.tex", plot10tpr)

# image data
nd = 40
img_model_names = ["aae", "gano", "skip", "vae", "wae", "\\textcolor{gray}{knn}", 
"\\textcolor{gray}{osvm}", "fano", "fmgn", "dsvd", "vaek", "vaeo"]
img_model_ranks = [3.0, 6.2, 11.8, 6.0, 8.2, 8.0, 7.2, 5.8, 3.0, 6.2, 5.0, 2.5]
plot5img = ranks2tikzcd(img_model_ranks,img_model_names, 
	nemenyi_cd(length(img_model_ranks), nd, 0.05))
string2file("images_auc_auc_cdd5.tex", plot5img)

plot10img = ranks2tikzcd(img_model_ranks,img_model_names, 
	nemenyi_cd(length(img_model_ranks), nd, 0.1))
string2file("images_auc_auc_cdd10.tex", plot10img)

plot10img = ranks2tikzcd(img_model_ranks,img_model_names, 
	nemenyi_cd(length(img_model_ranks), 4, 0.1))
string2file("images_auc_auc_cdd10_4_datasets.tex", plot10img)


# img data per class
include("crit_diag_functions.jl")
nd = 40
img_model_names = ["aae", "gano", "skip", "vae", "wae", "\\textcolor{gray}{knn}", 
"\\textcolor{gray}{osvm}", "fano", "fmgn", "dsvd", "vaek", "vaeo"]
img_model_ranks_per_class = [5.0, 6.8, 10.6, 5.8, 6.2, 6.3, 7.4, 6.4, 
	4.6, 6.5, 3.9, 3.2]
plot10img = ranks2tikzcd(img_model_ranks_per_class,img_model_names, 
	nemenyi_cd(length(img_model_ranks_per_class), nd, 0.1))
string2file("images_auc_auc_cdd10_per_class.tex", plot10img)
