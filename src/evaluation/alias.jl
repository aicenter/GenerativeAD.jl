using DataFrames

const MODEL_MERGE = Dict{String, String}(
	"aae_vamp" 	=> "aae",
	"wae_vamp" 	=> "wae",
	"vae+ocsvm" => "vae_ocsvm"
)

const MODEL_TYPE = Dict{String, String}(
	"MAF" 			=> "flows",
	"RealNVP"		=> "flows",
	"sptn"			=> "flows",
	"MO_GAAL" 		=> "gans",
	"fmgan" 		=> "gans",
	"gan" 			=> "gans", 
	"abod" 			=> "classical",
	"hbos" 			=> "classical",
	"if" 			=> "classical",
	"knn" 			=> "classical",
	"loda" 			=> "classical",
	"lof" 			=> "classical", 
	"ocsvm" 		=> "classical",
	"ocsvm_rbf"		=> "classical",
	"ocsvm_n1"		=> "classical",
	"pidforest" 	=> "classical", 
	"GANomaly" 		=> "autoencoders",
	"aae" 			=> "autoencoders",
	"aae_vamp" 		=> "autoencoders",
	"adVAE" 		=> "autoencoders", 
	"vae" 			=> "autoencoders",
	"wae" 			=> "autoencoders", 
	"wae_vamp" 		=> "autoencoders",
	"vae_simple"	=> "autoencoders",
	"vae_ocsvm" 	=> "two-stage",
	"vae_knn"	 	=> "two-stage",
	"DeepSVDD" 		=> "two-stage"
)

const MODEL_ALIAS = Dict{String, String}(
	"MAF" 				=> "maf",
	"RealNVP"			=> "rnvp",
	"sptn"				=> "sptn",
	"MO_GAAL" 			=> "mgal",
	"fmgan" 			=> "fmgn",
	"gan" 				=> "gan", 
	"abod" 				=> "abod",
	"hbos" 				=> "hbos",
	"if" 				=> "if",
	"knn" 				=> "knn",
	"loda" 				=> "loda",
	"lof" 				=> "lof", 
	"ocsvm" 			=> "osvm",
	"ocsvm_rbf"			=> "orbf",
	"ocsvm_n1"			=> "osnu",
	"pidforest" 		=> "pidf", 
	"GANomaly" 			=> "gano",
	"aae" 				=> "aae",
	"aae_vamp" 			=> "aaev",
	"adVAE" 			=> "avae", 
	"vae" 				=> "vae",
	"wae" 				=> "wae", 
	"wae_vamp" 			=> "waev",
	"Conv-GANomaly" 	=> "gano",
	"Conv-SkipGANomaly" => "skip",
	"vae_simple" 		=> "vaes",
	"vae_ocsvm" 		=> "vaeo",
	"vae_knn"			=> "vaek",
	"DeepSVDD"			=> "dsvd"
)

const DATASET_ALIAS = Dict{String, String}(
	"abalone"					=> "aba",
	"annthyroid"				=> "ann",
	"arrhythmia"				=> "arr",
	"blood-transfusion"			=> "blt",
	"breast-cancer-wisconsin"	=> "bcw",
	"breast-tissue"				=> "bts",
	"cardiotocography"			=> "crd",
	"ecoli"						=> "eco",
	"glass"						=> "gls",
	"haberman"					=> "hab",
	"har"						=> "har",
	"htru2"						=> "htr",
	"ionosphere"				=> "ion",
	"iris"						=> "irs",
	"isolet"					=> "iso",
	"kdd99_small"				=> "kdd",
	"letter-recognition"		=> "ltr",
	"libras"					=> "lbr",
	"magic-telescope"			=> "mgc",
	"mammography"				=> "mam",
	"miniboone"					=> "mnb",
	"multiple-features"			=> "mlt",
	"page-blocks"				=> "pgb",
	"parkinsons"				=> "prk",
	"pendigits"					=> "pen",
	"pima-indians"				=> "pim",
	"seismic"					=> "sei",
	"sonar"						=> "snr",
	"spambase"					=> "spm",
	"spect-heart"				=> "sph",
	"statlog-satimage"			=> "sat",
	"statlog-segment"			=> "seg",
	"statlog-shuttle"			=> "sht",
	"statlog-vehicle"			=> "vhc",
	"synthetic-control-chart"	=> "scc",
	"wall-following-robot" 		=> "wrb",
	"waveform-1"				=> "wf1",
	"waveform-2"				=> "wf2",
	"wine"						=> "wne",
	"yeast"						=> "yst",
	"MNIST" 					=> "mnist",
	"FashionMNIST" 				=> "fmnist",
	"CIFAR10" 					=> "cifar10",
	"SVHN2" 					=> "svhn2"
)

const AC_CONVERSION = Dict(
	"mnist" 			=> ["5", "0", "4", "1", "9", "2", "3", "6", "7", "8"],
	"svhn2"				=> ["1", "9", "2", "3", "5", "8", "7", "4", "6", "0"],
	"fmnist" 			=> ["Ankle boot", "T-Shirt", "Dress", "Pullover", "Sneaker", "Sandal", "Trouser", "Shirt", "Shirt", "Coat", "Bag"],
	"cifar10"			=> ["frog", "truck", "deer", "automobile", "bird", "horse", "ship", "cat", "dog", "airplane"]
)


"""
    apply_aliases!(df; col="modelname", d=MODEL_ALIAS)

Renames entries of column `col` according to dictionary `d`.
"""
function apply_aliases!(df::D; col="modelname", d=MODEL_ALIAS) where {D <: AbstractDataFrame}
    !(col in names(df)) && error("$col is not in the DataFrame columns.")
    df[:,col] = map(x -> get(d, x, x), df[:, col])
end


"""
	convert_anomaly_class(ac, dataset="MNIST")

Returns string representation of anomaly class id `ac`.
"""
function convert_anomaly_class(ac, dataset="mnist")
	getindex(AC_CONVERSION[dataset], ac)
end
