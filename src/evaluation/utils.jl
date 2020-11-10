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
collect_files(target) = collect_files!(target, [])