using PyCall

# loading a pymodel
function load_sgvae_model(dir, device)
    py"""
import sgad
from sgad.sgvae import SGVAE
from sgad.utils import load_model

def model(dir, device):
    return load_model(SGVAE, dir, device=device)
    """

    return py"model"(dir, device)
end

function load_cgn_model(dir, device)
    py"""
import sgad
from sgad.utils import load_cgnanomaly

def model(dir, device):
    return load_cgnanomaly(dir, device=device)
    """

    return py"model"(dir, device)
end

function load_sgvaegan_model(dir, device)
    py"""
import sgad
from sgad.sgvae import SGVAEGAN
from sgad.utils import load_model

def model(dir, device):
    return load_model(SGVAEGAN, dir, device=device)
    """

    return py"model"(dir, device)
end

function load_gan_model(dir, device)
    py"""
import sgad
from sgad.sgvae import GAN
from sgad.utils import load_model

def model(dir, device):
    return load_model(GAN, dir, device=device)
    """

    return py"model"(dir, device)
end

function load_vaegan_model(dir, device)
    py"""
import sgad
from sgad.sgvae import VAEGAN
from sgad.utils import load_model

def model(dir, device):
    return load_model(VAEGAN, dir, device=device)
    """

    return py"model"(dir, device)
end

# this is for fitting the logistic regression
mutable struct LogReg
	alpha
end

LogReg() = LogReg(nothing)

function fit!(lr::LogReg, X, y)
	py"""
from sgad.sgvae.utils import logreg_fit

def fit(X,y):
	alpha, _ = logreg_fit(X, 1-y)
	return alpha
	"""
	lr.alpha = py"fit"(X, y)
end

function predict(lr::LogReg, X; kwargs...)
	py"""
from sgad.sgvae.utils import logreg_prob

def predict(X, alpha):
	return logreg_prob(X, alpha)
	"""
	return py"predict"(X, lr.alpha)
end

# this is for fitting the softmax logistic regression
mutable struct ProbReg
	ac
	alpha
end

function ProbReg()
	py"""
from sgad.sgvae import AlphaClassifier
	"""
	m = ProbReg(py"AlphaClassifier()", nothing)
	return m
end

function fit!(m::ProbReg, X, y; scale=true, kwargs...)
	m.ac.fit(X, y; scale=scale, kwargs...)
	m.alpha = m.ac.get_alpha().detach().numpy()
end

function predict(m::ProbReg, X; scale=true, kwargs...)
	if scale
		X = m.ac.scaler_transform(X)
	end
	scores = vec(m.ac(X).detach().numpy())
end

# this is for fitting the robust logistic regression
mutable struct RobReg
    lr
    alpha
end

function RobReg(;input_dim=4, alpha=[1,1,1,1], alpha0=[1,0,0,0], beta=1.0)
	py"""
from sgad.sgvae import RobustLogisticRegression
	"""
	m = RobReg(py"RobustLogisticRegression(input_dim=$(input_dim), alpha=$(alpha), beta=$(beta),alpha0=$(alpha0))", nothing)
	return m
end

function fit!(m::RobReg, X, y; scale=true, verb=false, early_stopping=true, patience=1, kwargs...)
	m.lr.fit(X, y; scale=scale, verb=verb, early_stopping=early_stopping, patience=patience, kwargs...)
	m.alpha = m.lr.get_alpha().detach().numpy()
end

function predict(m::RobReg, X; scale=true, kwargs...)
	if scale
		X = m.lr.scaler_transform(X)
	end
	scores = vec(m.lr.predict_prob(X).detach().numpy())
end

function dropnames(namedtuple::NamedTuple, names::Tuple{Vararg{Symbol}}) 
    keepnames = Base.diff_names(Base._nt_names(namedtuple), names)
    return NamedTuple{keepnames}(namedtuple)
end

np = pyimport("numpy")
torch = pyimport("torch")

denormalize(x) = (x ./ 2) .+ 0.5

_to_img(x, r=1) = imresize(hcat(map(i->GenerativeAD.Datasets.array_to_img_rgb(permutedims(x[:,:,:,i], (2,1,3)))
            , 1:size(x,4))...), ratio=r)

to_img(x, r=1; descale=false) = descale ? _to_img(denormalize(x), r) : _to_img(x, r)

jl2py(x::Array, device="cuda") = torch.tensor(np.array(Array(permutedims(x, [4,3,2,1])))).to(device)
py2jl(x) = permutedims(torch2np(x), [4,3,2,1])
torch2np(x) = x.detach().cpu().numpy()

clamp(x::AbstractArray{T,N},l::T,u::T) where T where N = max.(min.(x, u),l)