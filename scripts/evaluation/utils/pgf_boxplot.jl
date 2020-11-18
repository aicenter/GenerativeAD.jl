_pgf_array(a) = "{$(join(a, ','))}"

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