# from https://github.com/vitskvara/PaperUtils.jl

"""
    eol(s)
Replaces the "& " at the end of s with a tabular end of line.
"""
function eol(s)
    return string(s[1:end-2], " \\\\ \n")
end

"""
    wspad(s, n)
Pads s with n white spaces.
"""
function wspad(s, n)
    return string(s, repeat(" ", n))
end
function ranks2tikzcd(ranks, algnames, c, caption = ""; scale=1.0,  
    vpad=0.3, label = "", pos = "h", levelwidth=0.06, vdist=0.1)
    n = length(ranks)
    @assert n == length(algnames)
    ranks = reshape(ranks, n)

    # header
    s = "\\begin{tikzpicture}[scale=$(scale)] \n"
    s = wspad(s, 2)
    #axis
    s = string(s, drawaxis(ranks))
    # nodes
    s = string(s, drawnodes(ranks, algnames; vpad=vpad))
    # levels
    s = string(s, drawlevels(ranks, c; levelwidth=levelwidth, vdist=vdist))
    # end of figure
    s = wspad(s, 1)
    s = string(s, "\\end{tikzpicture} \n")
    
    return s
end

function drawaxis(ranks)
    mx = ceil(maximum(ranks))
    mn = floor(minimum(ranks))

    s = "\\draw ($(mn),0) -- ($(mx),0); \n"
    s = wspad(s, 2)
    s = string(s, "\\foreach \\x in {$(Int(mn)),...,$(Int(mx))} \\draw (\\x,0.10) -- (\\x,-0.10) node[anchor=north]{\$\\x\$}; \n")
    return s
end

function drawnodes(ranks, algnames; vpad=0.3)
    isort = sortperm(ranks)
    ranks = ranks[isort]
    algnames = algnames[isort]

    mx = ceil(ranks[end])
    mn = floor(ranks[1])
    nor = length(ranks) # number of ranks
    nos = Int(ceil(nor/2)) # number of ranks on one side (start with left, there may be one more if nor is odd)

    s = ""
    for (i, r) in enumerate(ranks)
        s = wspad(s,2)
        if i<=nos
            s = string(s, "\\draw ($(r),0) -- ($r,$(i*vpad-0.1)) -- ($(mn-0.1), $(i*vpad-0.1)) node[anchor=east] {$(algnames[i])}; \n")
        else
            s = string(s, "\\draw ($(r),0) -- ($r,$((nor-i)*vpad+0.2)) -- ($(mx+0.1), $((nor-i)*vpad+0.2)) node[anchor=west] {$(algnames[i])}; \n")
        end
    end

    return s
end

function drawlevels(ranks, cv; levelwidth=0.06, vdist=0.1)
    ranks = sort(ranks)

    mx = ceil(ranks[end])
    mn = floor(ranks[1])
    nor = length(ranks) # number of ranks

    # do this differently
    ranks = sort(ranks)

    mx = ceil(ranks[end])
    mn = floor(ranks[1])
    nor = length(ranks) # number of ranks

    # do a matrix that says which ranks are connected
    con_mat = zeros(nor, nor)
    for i in 1:nor
        for j in 1:nor
            con_mat[i,j] = abs(ranks[j] - ranks[i]) < cv
        end
    end

    # now 
    levels = [] # each element is a tuple containing the (start, end, height) values
    lastlasti = 0
    for i in 1:nor
        if length(levels) == 0
            global levelh = vdist
        elseif lastlasti < i
            global levelh = vdist
        else
            lastlevelh = levels[end][end]
            if lastlevelh == vdist
                global levelh = lastlevelh + vdist
            else
                global levelh = lastlevelh + vdist
            end
        end
        startl = ranks[i]
        lasti = findlast(con_mat[i,:] .== 1)
        if lasti != nothing && lasti != lastlasti && lasti != i
            endl = ranks[lasti]
            push!(levels, (startl, endl, levelh))
        end
        lastlasti = lasti
    end

    # draw them
    s = ""
    for level in levels
        s = wspad(s,2)
        s = string(s, "\\draw[line width=$(levelwidth)cm,color=black,draw opacity=1.0] ($(level[1]-0.03),$(level[3])) -- ($(level[2]+0.03),$(level[3])); \n")
    end

    return s
end

"""
    string2file(f, s)
Save string s to file f.
"""
function string2file(f, s)
    open(f, "w") do _f
        write(_f, s)
    end
end