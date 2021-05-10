using Base.Threads: @threads

function collect_files!(target::String, files)
	if isfile(target)
		push!(files, target)
	else
		for file in readdir(target, join=true)
			collect_files!(file, files)
		end
	end
	files
end


"""
	collect_files(target)

Walks recursively the `target` directory, collecting all files only along the way.
"""
collect_files(target) = collect_files!(target, String[])


"""
	collect_files_th(target)

Multithreaded version of recursive file collection. 
Does not have as many checks as th single threaded.
May not perform as well on some configurations.
Using @threads inside recursion may not be the best idea.
"""
function collect_files_th(target::String)
    files = readdir(target, join=true)
    if all(isfile.(files))
        println(target)
        return files
    end
    results = Vector{Vector{String}}(undef, length(files))
    @threads for i in 1:length(files)
        # ignore any file that lives alongside a folder
        if !isfile(files[i])
            results[i] = collect_files_th(files[i])
        else
            results[i] = String[]
        end
    end
    reduce(vcat, results)
end
