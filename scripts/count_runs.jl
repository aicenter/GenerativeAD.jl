using DataFrames
using BSON, FileIO, ValueHistories
using PrettyTables.Crayons
using ArgParse
s = ArgParseSettings()
@add_arg_table s begin
    "modelpath"
    	required = true
    	arg_type = String
        help = "the path where model results are saved"
    "--check"
        help = "check validity of saved files, may take some time"
		action = :store_true
	"-n", "--min_samples"
        help = "Highlight folders where there are less than `min_samples` exp. files."
		arg_type = Int
		default = 100
end
parsed_args = parse_args(ARGS, s)
modelpath = parsed_args["modelpath"]
check = parsed_args["check"]
min_samples = parsed_args["min_samples"]

function list_files(datapath)
	fs = readdir(datapath)
	fs = filter(f->!(startswith(f, "model_")), fs)
	fs = filter(f->!(occursin("#", f)), fs)
	return unique(fs)
end
function _check_validity(file)
	data = load(file)
	(length(data[:val_labels]) == length(data[:val_scores])) && 
		length(data[:tst_labels]) == length(data[:tst_scores]) &&
		!any(isnan.(data[:val_scores])) &&
		!any(isnan.(data[:tst_scores]))
end
function check_model_dataset(modelpath, check_validity=true, min_samples=100)
	modelname = basename(modelpath)
	@info "Scanning model $modelname"
	datasetpaths = joinpath.(modelpath, readdir(modelpath))
	invalid_files = []
	for datasetpath in datasetpaths
		dataset = basename(datasetpath)
		println("")
		println(Crayon(foreground=(255,255,255)), "DATASET = $dataset")
		seedpaths = joinpath.(datasetpath, readdir(datasetpath))
		for seedpath in seedpaths
			seed = basename(seedpath)[end]
			ufiles = joinpath.(seedpath,list_files(seedpath))
			nuf = length(ufiles)
			col = (length(ufiles) < min_samples) ? (255, 153, 51) : (255, 255, 255)
			if check_validity
				if length(ufiles) == 0
					nvalid = 0
				else
					valid_inds = _check_validity.(ufiles)
					invalid_files = vcat(invalid_files, ufiles[.!valid_inds])
					nvalid = sum(valid_inds)
				end
				col = (nvalid < nuf) ? (255,0,0) : col
				println(Crayon(foreground=col), "SEED = $seed: $(nvalid)/$(nuf)")
			else
				println(Crayon(foreground=col), "SEED = $seed: $(nuf)")
			end
		end
	end
	return invalid_files
end

invalid_files = check_model_dataset(modelpath, check, min_samples)
# without this, bash color may remain changed
println(Crayon(foreground=(255,255,255)), "\nDONE")
