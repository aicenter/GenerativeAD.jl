using Statistics
using Distributions
using DrWatson
@quickactivate
include("crit_diag_functions.jl")

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
nm = 24
α = 0.05
ncv5 = nemenyi_cd(nm, nd, α) # = 5.75
α = 0.1
ncv10 = nemenyi_cd(nm, nd, α) # = 5.40

nd = 40
#
model_names = ["aae", "avae", "gano", "vae", "wae", "abod", "hbos", "if", "knn", "loda", "lof", "osvm", 
"pidf", "maf", "rnvp", "sptn", "fmgn", "gan", "mgal", "dagm", "dsvd", "rpn", "vaek", "vaeo"]
model_names_gray = ["aae", "avae", "gano", "vae", "wae", "\\textcolor{gray}{abod}", 
"\\textcolor{gray}{hbos}", "\\textcolor{gray}{if}", "\\textcolor{gray}{knn}", "\\textcolor{gray}{loda}", 
"\\textcolor{gray}{lof}", "\\textcolor{gray}{osvm}", "\\textcolor{gray}{pidf}", 
"maf", "rnvp", "sptn", "fmgn", "gan", "mgal", "dagm", "dsvd", "rpn", "vaek", "vaeo"]
model_ranks = [6.9, 13.4, 10.3, 6.5, 7.0, 11.4, 14.9, 14.5, 8.4, 16.1, 11.4, 2.9, 14.3, 8.9, 9.0, 10.7, 
	11.4, 12.0, 22.3, 19.8, 16.2, 12.1, 11.0, 6.8]

plot5 = ranks2tikzcd(model_ranks, model_names_gray, 
	nemenyi_cd(length(model_ranks), nd, 0.05); 
	scale=0.45, vpad=0.6, levelwidth=0.03, vdist=0.15)
string2file(datadir("paper_results/tabular_auc_auc_cdd5.tex"), plot5)

plot10 = ranks2tikzcd(model_ranks, model_names_gray, 
	nemenyi_cd(length(model_ranks), nd, 0.1); 
	scale=0.45, vpad=0.6, levelwidth=0.03, vdist=0.15)
string2file(datadir("paper_results/tabular_auc_auc_cdd10.tex"), plot10)

# clean tabular
model_ranks = [12.5, 16.2, 11.6, 10.4, 9.0, 8.6, 11.3, 8.6, 6.0, 18.3, 8.0, 6.0, 12.5, 7.2, 7.1, 7.4, 
14.8, 15.4, 20.6, 19.7, 16.6, 19.5, 10.7, 11.6]
plot10 = ranks2tikzcd(model_ranks, model_names_gray, 
	nemenyi_cd(length(model_ranks), nd, 0.1); 
	scale=0.65, vpad=0.4, levelwidth=0.03, vdist=0.13)
string2file(datadir("paper_results/tabular_clean_auc_auc_cdd10.tex"), plot10)

# image data - statistic
img_model_names = ["aae", "gano", "skip", "vae", "wae", "\\textcolor{gray}{knn}", 
"\\textcolor{gray}{osvm}", "fano", "fmgn", "dsvd", "vaek", "vaeo"]
img_model_ranks = [4.0, 9.5, 10.5, 2.7, 2.5, 5.3, 6.0, 3.5, 4.1, 6.9, 4.8, 4.1]

nd = 19
nm = length(img_model_names)
plot5img = ranks2tikzcd(img_model_ranks,img_model_names, 
	nemenyi_cd(length(img_model_ranks), nd, 0.05))
string2file(datadir("paper_results/images_auc_auc_cdd5.tex"), plot5img)

plot10img = ranks2tikzcd(img_model_ranks,img_model_names, 
	nemenyi_cd(length(img_model_ranks), nd, 0.1))
string2file(datadir("paper_results/images_auc_auc_cdd10.tex"), plot10img)

ncv10 = nemenyi_cd(nm, nd, 0.1) # = 3.54

nd = 37
img_model_ranks = [3.6, 5.1, 9.1, 3.1, 2.9, 5.4, 6.4, 3.4, 7.5, 4.0, 5.7, 4.9]
nm = length(img_model_names)
ncv10 = nemenyi_cd(nm, nd, 0.1) # = 2.54
plot10img = ranks2tikzcd(img_model_ranks,img_model_names, ncv10)
string2file(datadir("paper_results/images_auc_auc_cdd10_granular.tex"), plot10img)

# statistic - clean
img_model_ranks = [3.2, 5.2, 9.1, 3.6, 4.1, 3.8, 5.7, 6.0, 10.1, 7.5, 4.7, 5.2]
plot10img = ranks2tikzcd(img_model_ranks,img_model_names, 
	nemenyi_cd(length(img_model_ranks), nd, 0.1))
string2file(datadir("paper_results/images_clean_auc_auc_cdd10_granular.tex"), plot10img)

# image data - semantic
img_model_names = ["aae", "gano", "skip", "vae", "wae", "\\textcolor{gray}{knn}", 
"\\textcolor{gray}{osvm}", "fano", "fmgn", "dsvd", "vaek", "vaeo"]
img_model_ranks = [6.5, 10.0, 12.0, 4.0, 5.0, 9.5, 8.0, 4.5, 1.0, 6.0, 4.0, 5.5]

nd = 2
plot5img = ranks2tikzcd(img_model_ranks,img_model_names, 
	nemenyi_cd(length(img_model_ranks), nd, 0.05))
string2file(datadir("paper_results/images_auc_auc_cdd5_semantic.tex"), plot5img)

plot10img = ranks2tikzcd(img_model_ranks,img_model_names, 
	nemenyi_cd(length(img_model_ranks), nd, 0.1))
string2file(datadir("paper_results/images_auc_auc_cdd10_semantic.tex"), plot10img)

ncv10 = nemenyi_cd(12, 2, 0.1) # = 10.92

img_model_ranks = [5.0, 8.1, 7.8, 3.8, 4.9, 7.8, 8.3, 4.8, 2.4, 6.2, 5.6, 6.6]
nd = 20
nm = length(img_model_ranks)
ncv10 = nemenyi_cd(nm, nd, 0.1) # = 3.45
plot10img = ranks2tikzcd(img_model_ranks,img_model_names, ncv10)
string2file(datadir("paper_results/images_auc_auc_cdd10_semantic_granular.tex"), plot10img)

# semantic - clean
img_model_ranks = [5.4, 5.2, 8.8, 7.0, 7.2, 4.8, 4.6, 6.6, 9.4, 3.4, 4.7, 4.8]
nd = 20
nm = length(img_model_ranks)
ncv10 = nemenyi_cd(nm, nd, 0.1) # = 3.45
plot10img = ranks2tikzcd(img_model_ranks,img_model_names, ncv10)
string2file(datadir("paper_results/images_clean_auc_auc_cdd10_semantic_granular.tex"), plot10img)

# diff oocsvm - vae
difff = 6.5 - 2.9 # 3.6
nemenyi_cd(24, 82, 0.1)

# this is all outdated
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

# TODO also for more?
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
