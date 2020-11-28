_pgf_array(a) = "{$(join(a, ','))}"

const COLORS = ["cyan", "red", "orange", "lime", "magenta", "lightgray", "darkgray"]

function pgf_boxplot(lqs, hqs, mds, mns, mxs, labels; h="10cm", w="6cm")
	s = """
	\\begin{tikzpicture}
	\\begin{axis}[
		height=$h,
		width=$w,
		ymin=0, ymax=$(length(labels)+1),
		xmin=0, xmax=$(length(labels)+1),
		ytick=$(_pgf_array(1:length(labels))),
		yticklabels=$(_pgf_array(labels))
		]
	"""

	for (lq, hq, md, mn, mx) in zip(lqs, hqs, mds, mns, mxs)
		s *= """
		\\addplot [boxplot prepared={
		lower whisker=$mn, lower quartile=$lq,
		median=$md, upper quartile=$hq,
		upper whisker=$mx},
		] coordinates {};
		"""
	end
	s *= """
	\\end{axis}
	\\end{tikzpicture}
	"""
	s
end

function pgf_boxplot_grouped(lqs, hqs, mds, mns, mxs, labels, groups; h="10cm", w="6cm")
	s = """
	\\begin{tikzpicture}
	\\begin{axis}[
		height=$h,
		width=$w,
		ymin=0, ymax=$(length(labels)+1),
		xmin=0, xmax=$(length(labels)+1),
		ytick=$(_pgf_array(1:length(labels))),
		yticklabels=$(_pgf_array(labels))
		]
	"""

	for (lq, hq, md, mn, mx, g) in zip(lqs, hqs, mds, mns, mxs, groups)
		s *= """
		\\addplot [$(g%2==0 ? "fill=lightgray" : "fill=gray"), boxplot prepared={
		lower whisker=$mn, lower quartile=$lq,
		median=$md, upper quartile=$hq,
		upper whisker=$mx},
		] coordinates {};
		"""
	end
	s *= """
	\\end{axis}
	\\end{tikzpicture}
	"""
	s
end

function pgf_boxplot_grouped_color(lqs, hqs, mds, mns, mxs, labels, groups; h="10cm", w="6cm")
	s = """
	\\begin{tikzpicture}
	\\begin{axis}[
		height=$h,
		width=$w,
		ymin=0, ymax=$(length(labels)+1),
		xmin=0, xmax=$(length(labels)+1),
		ytick=$(_pgf_array(1:length(labels))),
		yticklabels=$(_pgf_array(labels))
		]
	"""

	for (lq, hq, md, mn, mx, g) in zip(lqs, hqs, mds, mns, mxs, groups)
		s *= """
		\\addplot [fill=$(COLORS[g]), boxplot prepared={
		lower whisker=$mn, lower quartile=$lq,
		median=$md, upper quartile=$hq,
		upper whisker=$mx},
		] coordinates {};
		"""
	end
	s *= """
	\\end{axis}
	\\end{tikzpicture}
	"""
	s
end


function pgf_boxplot_grouped_colorpos(lqs, hqs, mds, mns, mxs, labels, groups, mgroups; h="10cm", w="6cm")
	yticks = [1]
	for (cg, ng) in zip(mgroups[1:end-1], mgroups[2:end-1])
		if ng == cg
			push!(yticks, yticks[end] + 1)
		else
			push!(yticks, yticks[end] + 2)
		end
	end
	push!(yticks, yticks[end] + 1)


	s = """
	\\begin{tikzpicture}
	\\begin{axis}[
		height=$h,
		width=$w,
		ymin=0, ymax=$(yticks[end] + 1),
		xmin=0, xmax=$(length(labels) + 1),
		ytick=$(_pgf_array(yticks)),
		yticklabels=$(_pgf_array(labels))
		]
	"""

	for (lq, hq, md, mn, mx, g, y) in zip(lqs, hqs, mds, mns, mxs, groups, yticks)
		s *= """
		\\addplot [fill=$(COLORS[g]), boxplot prepared={
		draw position=$(y),
		lower whisker=$mn, lower quartile=$lq,
		median=$md, upper quartile=$hq,
		upper whisker=$mx},
		] coordinates {};
		"""
	end
	s *= """
	\\end{axis}
	\\end{tikzpicture}
	"""
	s
end