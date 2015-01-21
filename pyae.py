import numpy as np
import theano
import theano.tensor as T


class AutoencoderPuppet:
	""" Basic build class for any AE """
	def __init__(self, inp):
		self.inp = inp

	def encode(self, value):
		pass


	def decode(self, value):
		pass

	def _cost(self, value):
		pass

	def _get_grads(self):
		pass

	def train(self):
		pass



class Loss:
	def __init__(self):
		pass

	def CrossEntropy(self,x, y):
		return T.sum(T.dot(x, y)  + T.dot(1 - x, T.log(1 - y)))

	def Hinge(self, x):
		return T.max(0, 1 + x)

	def ReLU(self, x):
		return T.max(0, x)



class Autoencoder(AutoencoderPuppet):
	def __init__(self, x, num_vis=100, num_hid=50):
		self.x = x
		numpy_rng = np.random.RandomState()
		self.W = theano.shared(value = \
			numpy.asarray(numpy_rng.uniform(low=0.0, high=1.0, size=((num_vis,num_hidd)), dtype=theano.config.floatX)), name='W')
		self.b = theano.shared(numpy.asarray(numpy_rng.uniform(low=0.0, high=1.0, size=(num_hidd,), dtype=theano.config.floatX)), name='b')
		self.params = [self.W, self.b]

	def _cost(self, activefunc):
		hidden = self.encoder(value, self.W)
		#Transpose version of W
		decode = self.encoder(hiddenself.W.T)
		L = -T.sum(hidden * T.log(decode) + (1 - hidden) * T.log(1 - decode))
		return T.mean(L)

	def _get_grads(self, cost):
		return T.grad(cost, self.params)

	def encoder(self, value, W):
		return T.nnet.sigmoid(T.dot(W, value) + self.b)

	def train(self, batch_size=0.2, activation='sigmoid', iters=5000, lrate=0.05):
		""" 
			batch_size in percent, how many data choice from dataset on each iteration
		"""

		activfunc = T.nnet.sigmoid
		if activation == 'tanh':
			activfunc == T.tanh

		for i in range(iters):
			cost = self._cost()
			grads = self._get_grads(cost)

			#Update params
			updates=[(param, param - lrate * newparam) for param, newparam in zip(self.params, grads)]

	def _innerTrain(self, data):
		pass


	def predict(self, newvalue):
		""" Predict new data after training """
		pass



class DenoisingAutoencoder(AutoencoderPuppet):
	def __init__(self, x):
		AutoencoderPuppet.__init__(self, x)

	def train(self, activation='sigmoid', corruption=0.4):
		pass


class StackedAutoencoder(AutoencoderPuppet):
	""" 
	http://ufldl.stanford.edu/wiki/index.php/Stacked_Autoencoders 
	"""
	def __init__(self, x, layers=3, corruption_level):
		"""
			|corruption_level| = layers
		"""
		AutoencoderPuppet.__init__(self, x)
		self.layers = layers
		self.corruption_level = corruption_level
		self.layers = [Autoencoder(self.x) for i in self.layers]

	def train(self, layers=3, iters=5000):
		"""
			layers - number of DA
		"""
		pass



class StackedConvolutionalAutoencoder(AutoencoderPuppet):
	"""
		http://people.idsia.ch/~ciresan/data/icann2011.pdf
	"""
	def __init__(self, x):
		AutoencoderPuppet.__init__(self, x)


class MarginalizedDenoisingAutoencoder(AutoencoderPuppet):
	"""
		http://arxiv.org/pdf/1206.4683v1.pdf
	"""
	def __init__(self, x):
		AutoencoderPuppet.__init__(self, x)

	def _appendBias(value):
		newvalue = T.matrix('xb')

	def _cost(self, p, activation='tanh'):
		if activation == 'sigmoid':

		W = T.matrix('W')
		self.x = self._appendBias(self.x)
		value = np.ones((d, 1)) * (1 - p)
		value[d-1] = 1
		corrm = T.shared(np.dot(value, value.T))
		prod = T.dot(self.x, self.x)
		Q = T.dot(prod, corrm)
		R = np.dot(value, T.diag(prod))
		rep = T.dot(prod, np.repeat(value, d))
		return T.tanh(np.dot(self.x, W))

	def _get_grads(self, cost):
		return T.grad(self._cost, self.params)

	def train(self, p, iters=5000):
		for i in range(iters):
			cost = self._cost(p)
			grads = self._get_grads(cost)


class Model:
	""" Basic building block for running autoencoders, howewer
		classes for autoencoders can be use without this class
	 """
	def __init__(self, *args, **kwargs):
		self.inp = kwargs.get('input')
		if self.inp == None:
			raise Exception("Can't find input for learning")

	def addDataset(self, path):
		pass

	def add(self, title, aemodel):
		if not isinstance(aemodel, AutoencoderPuppet):
			raise Exception("This model is not from AutoencoderPuppet class")

