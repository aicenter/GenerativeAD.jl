"""
	function [H,LVpen] = histoptimal(Y, Dmax)
 PERFORMS DENSITY ESTIMATION using PENALIZED REGULAR HISTOGRAM, author Yves ROZENHOLC
 Compute automaticaly the number of bins of the classical density estimator, 
 the regular histogram, and return description of the obtained histogram(s). The 
 number(s) of bins chose by the algorithm derived from the works of 
 MODEL and COMMANDE LINE 
 MODEL : 
 Y n-sample of an UNKNOWN density f 
 COMMANDE LINE : 
 H = histoptimal(Y, Dmax, withfig)
 [H,LVpen] = histoptimal(Y, Dmax, withfig)
 INPUT
      Y, the observations - an n sample of an unknown density f
      Dmax, the maximum number of bins of the histogram
           Default value [n/log(n)].
           For very large n, use smaller value to reduce the computation time.
      withfig, if 1 plot the selected histogram 
           Default 1.
 OUTPUT
      H, the selected histogram 
      LVpen, the penalized Log-Likelihood of the selected histogram
 CONNECTED PAPERS 
 HOW MANY BINS SHOULD BE PUT IN A REGULAR HISTOGRAM.
 by L. BIRG\'E and Y. ROZENHOLC in ESAIM P&S
 Other references :
 	A. Barron, L. Birg'e, P.Massart 1995 
 	G. Castellan 1998 
 for details you can contact
 	Yves ROZENHOLC
 	yves.rozenholc@univ-paris5.fr
 version du 17/03/00
"""
function histoptimal(Y, Dmax::Int = floor(Int, length(Y)/log(length(Y))))
	Y = Y[:];
	n = length(Y)
	D = 2:Dmax;
	pen = (D .- 1) .+ log.(D) .^ 2.5;

	mini, maxi =minimum(Y), maximum(Y);
	mini ≈ maxi && @error "almost zero range of data"
	if (mini<0)||(maxi>1)
		Y = (Y .- mini)./(maxi - mini); 
	else
   mini, maxi = 0, 1;
	end;

	LV = zeros(length(D)); #Le vecteur des max de la log-vraisemblance
	for (i, d) = enumerate(D)
		Nd = bindata(Y, d); # valeur de la fct de repartition aux bornes
		ind = findall(Nd .!=0 );
    LV[i] = sum(Nd[ind] .* log.(Nd[ind] .* d ./ n));
	end
	ii = argmax(LV .- pen)
	LV[ii] - pen[ii], D[ii] 
end

function bindata(x, k::Int)
	δ = (maximum(x) - minimum(x)) / k
	(δ == 0) && @error "cannot bin interval of zero length"
	o = zeros(k);
	bindata!(o, x, δ)
end

function bindata(x, δ::AbstractFloat)
	o = zeros(ceil(Int,(maximum(x) - minimum(x)) / δ))
	bindata!(o, x, δ)
end

function bindata!(o, x, δ::AbstractFloat)
	foreach(z -> o[min(length(o),floor(Int, z/δ) + 1)] += 1, x)
	o 
end

struct EqualHistogram{T}
	p :: Vector{T}
	mi::T
	mx::T
	δ::T
end

function EqualHistogram(x)
	mi, mx = minimum(x), maximum(x)
	mx - mi == 0 && @error("The span of histogram should be greater than zero")

	x = (x .- mi) ./ (mx - mi)

	k = histoptimal(x)[2]
	δ = 1 / k
	p = bindata(x, δ) ./ (length(x)*(mx - mi))
	EqualHistogram(p, mi, mx, δ)
end

function (m::EqualHistogram)(x)
	ii = floor.(Int, (x .- m.mi) ./ ((m.mx - m.mi) * m.δ))
	o = similar(x)
	fill!(o, log(10*eps(eltype(x))))
	mask = (ii .>0 ) .& (ii .< length(m.p))
	o[mask] .= -log.( m.p[ii[mask]] .+ 10*eps(eltype(x)))
	o
end


struct Loda{V,E}
	ws::Vector{V}
	histograms::Vector{E}
end

criterion(fs) = sum(abs.(fs[end] .- fs[end - 1])) / sum(abs.(fs[2] .- fs[1]))

function Loda(x, τ::AbstractFloat = 0.01, sparse::Bool = false)
	sparse == true && @error("Not implemented")
	Loda(optimize_loda_params(x, τ)[1:2]...)
end

function optimize_loda_params(x,τ)
	d = size(x,1)
	ws = []
	fs = []
	histograms = []
	k = 1
	cumf = zeros(size(x,2))
	while true
		w = randn(1, d)
		p = w * x
		h = EqualHistogram(p)
		cumf .+= h(p)[:]
		push!(fs, cumf ./ k)
		push!(ws, w)
		push!(histograms, h)
		
		length(fs) > 2 && criterion(fs) < τ && break
		k += 1 
	end
	(ws, histograms, fs)
end

function (m::Loda)(x)
	o = similar(x, size(x,2))
	o .= 0
	for i in 1:length(m.ws)
		h, w = m.histograms[i], m.ws[i]
		o .-= h(w * x)[:]
	end
	o ./= length(m.ws)
end