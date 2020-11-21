import torch 
import torch.nn as nn 
import torch.nn.functional as F
import numpy as np
import torch.utils.data as utils


class Flatten(nn.Module):
	def __init__(self, out_features):
		super(Flatten, self).__init__()
		self.output_dim = out_features

	def forward(self, x):
		return x.view(-1, self.output_dim)


class Reshape(nn.Module):
	def __init__(self, out_shape):
		super(Reshape, self).__init__()
		self.out_shape = out_shape

	def forward(self, x):
		return x.view(-1, *self.out_shape)


def encoder_builder(idim, zdim, kernelsizes=[9,7,5,3], channels=[128,64,32,16], scalings=[2,2,2,1], 
					activation=nn.ReLU(), batchnorm=False):
	nconv = len(kernelsizes)
	if len(kernelsizes) != len(scalings):
		raise ValueError("incompatible input dimensions")
	if len(idim) != 3:
		raise ValueError("idim must be (c,h,w)")

	# check
	h0 = idim[1]/np.prod(scalings)
	w0 = idim[2]/np.prod(scalings)

	if ((h0 != int(np.floor(h0))) or (w0 != int(np.floor(w0)))):
		raise ValueError("your input size and scaling is not compatible")

	encoder = []
	cins = [idim[0]]+channels[:-1]

	for (k,ci,co,sc) in zip(kernelsizes, cins, channels, scalings):
		encoder.append(nn.Conv2d(ci, co, k, padding=int((k-1)/2)))
		encoder.append(activation)
		encoder.append(nn.MaxPool2d(sc))
		if batchnorm:
			encoder.append(nn.BatchNorm2d(co)) 
	
	encoder.append(Flatten(int(h0*w0*channels[-1])))
	encoder.append(nn.Linear(in_features=int(h0*w0*channels[-1]), out_features=zdim))
	return nn.Sequential(*encoder)


def decoder_builder(idim, zdim, kernelsizes=[3, 5, 7, 9], channels=[16, 32, 64, 128], scalings=[1,2,2,2], 
					activation=nn.ReLU(), batchnorm=False):
	nconv = len(kernelsizes)
	if len(kernelsizes) != len(scalings):
		raise ValueError("incompatible input dimensions")
	if len(idim) != 3:
		raise ValueError("idim must be (c,h,w)")
		
	h0 = idim[1]/np.prod(scalings)
	w0 = idim[2]/np.prod(scalings)

	if ((h0 != int(np.floor(h0))) or (w0 != int(np.floor(w0)))):
		raise ValueError("your input size and scaling is not compatible")
		
	couts = channels[1:]+[idim[0]]
	decoder = []
	decoder.append(nn.Linear(in_features=zdim, out_features=int(h0*w0*channels[0])))
	decoder.append(Reshape((channels[0], int(h0), int(w0))))
	
	for (k,ci,co,sc) in zip(kernelsizes, channels, couts, scalings):
		decoder.append(nn.ConvTranspose2d(ci, ci, sc, stride=sc))
		decoder.append(activation)
		decoder.append(nn.Conv2d(ci, co, k, padding=int((k-1)/2)))
		decoder.append(activation)
		if batchnorm:
			decoder.append(nn.BatchNorm2d(co)) 
	if batchnorm:
		return nn.Sequential(*decoder[:-2])
	else:
		return nn.Sequential(*decoder[:-1])
	

class SWISH(nn.Module):
	def __init__(self):
		super(SWISH, self).__init__()

	def forward(self, x):
		return x * torch.sigmoid(x)
  

class fAnoGAN(nn.Module):
	def __init__(
				self, 
				idim=(1,2,2), 
				zdim=100, 
				kernelsizes=[9,7,5,3], 
				channels=[128,64,32,16], 
				scalings=[2,2,2,1], 
				activation="relu",
				batchnorm=False,
				usegpu=True,
				**kwargs):
		
		super(fAnoGAN, self).__init__()
		if activation == "relu":
			activ_f = nn.ReLU()
		elif activation == "swish":
			activ_f = SWISH()
		elif activation == "tanh":
			activ_f = nn.Tanh()
		else:
			raise ValueError("unknown activation function")
		if usegpu:
			self.device = torch.device("cuda")
		else:
			self.device = torch.device("cpu")
		
		self.zdim = zdim
		self.encoder = encoder_builder(idim,zdim, kernelsizes, channels, scalings, activ_f, batchnorm).to(device)
		self.discriminator = encoder_builder(idim, 1, kernelsizes, channels, scalings, activ_f, batchnorm).to(device)
		kernelsizes.reverse()
		channels.reverse()
		scalings.reverse()
		self.generator = decoder_builder(idim,zdim, kernelsizes, channels, scalings, activ_f, batchnorm).to(device)
		

	def forward(self, x):
		"""
			izi_f forward pass
		""" 
		z = self.encoder(x)
		x_hat = self.generator(z)
		f = self.discriminator[:-1]
		fx = f(x)
		fx_hat = f(x_hat)
		return x_hat, fx, fx_hat


	def izif_loss(self, x, kappa=1.0):
		x_, fx, fx_ = self.forward(x)
		return F.mse_loss(x,x_) + kappa * F.mse_loss(fx, fx_)


	def gradient_penalty(self, x, x_, lambda_=10):
		bs = x.shape[0]
		dim = x.shape[2:4]
		alpha = torch.rand(bs, 1)
		alpha = alpha.expand(bs, int(x.nelement()/bs)).contiguous()
		alpha = alpha.view(x.shape).to(self.device)
		
		interpolates = alpha*x.detach() + ((1 - alpha) * x_.detach())
		interpolates = interpolates.to(self.device)
		interpolates.requires_grad_(True)

		disc_interpolates = self.discriminator(interpolates)  

		gradients = torch.autograd.grad(outputs=disc_interpolates, inputs=interpolates,
							  grad_outputs=torch.ones(
								  disc_interpolates.size()).to(self.device),
							  create_graph=True, retain_graph=True, only_inputs=True)[0]

		gradients = gradients.view(gradients.size(0), -1)
		gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean() * lambda_
		return gradient_penalty


	def predict(self, data, batch_size=64, kappa=1.0):
		self.generator.eval()
		self.discriminator.eval()
		self.encoder.eval()

		test_loader = utils.DataLoader(data[0], batch_size=batch_size)
		anomaly_scores = []
		for x in test_loader:
			x = x.to(self.device)
			x_, fx, fx_ = self.forward(x)
			L_G = torch.sum(torch.pow(x-x_,2), axis=[1,2,3]).detach().cpu()
			L_D = torch.sum(torch.pow(fx-fx_,2), axis=1).detach().cpu()
			anomaly_scores.append(L_G + kappa*L_D)
		anomaly_scores = torch.cat(anomaly_scores)
		return anomaly_scores


	def fit(self, data, max_iters=10000, lr_gan=1e-4, lr_enc=1e-4, batch_size=64, n_critic=5):
		one = torch.FloatTensor([1]).to(self.device)
		mone = one * -1

		optim_G = torch.optim.Adam(self.generator.parameters(), lr=lr_gan)
		optim_D = torch.optim.Adam(self.discriminator.parameters(), lr=lr_gan)
		optim_E = torch.optim.Adam(self.encoder.parameters(), lr=lr_enc)

		# data = (x, y)
		train_loader = utils.DataLoader(data[0], batch_size=batch_size, 
										sampler=utils.RandomSampler(
											torch.arange(len(data[1])),
											replacement=True, 
											num_samples=max_iters*batch_size)
										)

		history = {"generator": [], "discriminator": [], "encoder":[]}

		for iter, x in enumerate(train_loader):
			"""
				Train discriminator with adventage n_critic
			"""
			for i in range(n_critic):
				self.discriminator.zero_grad()

				real = x.to(self.device)
				D_real = self.discriminator(real)
				D_real = -D_real.mean()
				D_real.backward() # -E[discriminator(x)]

				noise = torch.randn(batch_size, self.zdim)
				noise = noise.to(self.device)
				fake = self.generator(noise).detach()
				D_fake = self.discriminator(fake)
				D_fake = D_fake.mean()
				D_fake.backward() # + E(discriminator(generator(z)))
				
				gradient_penalty = self.gradient_penalty(real.data, fake.data)
				gradient_penalty.backward()

				optim_D.step()
				history["discriminator"].append(D_fake.item() + D_real.item() + gradient_penalty.item())
				print("discriminator loss -> ", D_fake.item() + D_real.item() + gradient_penalty.item())

			"""
				Train generator
			"""
			self.generator.zero_grad()
			noise = torch.randn(batch_size, self.zdim)
			noise = noise.to(self.device)
			fake = self.generator(noise)
			G_loss = self.discriminator(fake)
			G_loss = - G_loss.mean()
			G_loss.backward()
			optim_G.step()
			history["generator"].append(G_loss.item())

			print(G_loss.item())

		"""
			Train encoder
		"""
		for iter, x in enumerate(train_loader):
			real = x.to(self.device)
			self.encoder.zero_grad()
			loss = self.izif_loss(real)
			loss.backward()
			optim_E.step()
			print(loss.item())
			history["encoder"].append(loss.item())

		return self
