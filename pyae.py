import numpy as np
import theano
import theano.tensor as T
from theano.tensor.shared_randomstreams import RandomStreams
from theano.printing import pprint
import pickle
from sklearn import datasets


class AutoencoderPuppet:
	""" Basic build class for any AE """
	def __init__(self, x=None, theano_input = None, num_vis=100, num_hid=50, numpy_rng=None, lrate=0.001, corruption_level=0, \
		encoder_func='sigmoid', decoder_func = None):
		'''
			x - input data in simple numpy ndarray format
			theano_input - input data in theano format (tensor.matrix, tensor.vector, etc..)
			num_vis - number of visible units
			num_hid - number of hidden units
			numpy_rng - generator for random numbers
			lrate - learning rate for training
			corruption_level - need to denoising autoencoders
			encoder_func - function for encoding
			decoder_func - function for decoding
			(bouth can be None)
		'''
		self.training = x
		if theano_input == None:
			self.x = T.matrix('x')
		else:
			self.x = theano_input
		self.x = T.matrix('x')
		if numpy_rng == None:
			numpy_rng = np.random.RandomState()
		par = ParametersInit(numpy_rng, -0.001, 0.001)
		self.W = par.get_weights(((num_vis, num_hid)))
		#Bias init as zero
		self.bh = theano.shared(np.asarray(np.zeros(num_hid), dtype=theano.config.floatX), name='bh')
		self.bv = theano.shared(np.asarray(np.zeros(num_vis), dtype=theano.config.floatX), name='bv')
		self.params = [self.W, self.bv, self.bh]
		self.lrate = lrate
		self.corruption_level = corruption_level
		self.num_vis = num_vis
		self.num_hid = num_hid
		self._set_encoder_func(encoder_func, decoder_func)


	def _set_encoder_func(self, encoder_func, decoder_func):
		if encoder_func == 'sigmoid':
			self._encoder_func = T.nnet.sigmoid
		if encoder_func == 'tanh':
			self._encoder_func = T.tanh
		else:
			self._encoder_func = T.nnet.sigmoid
		if decoder_func == 'sigmoid':
			self._decoder_func = T.nnet.sigmoid
		if decoder_func == 'tanh':
			self._decoder_func = T.tanh
		else:
			self._decoder_func =T.nnet.softmax

	'''def updateParams(self, cost):
		""" One step for update params
		"""
		grads = self._get_grads(cost)'''

	def setWeights(self, newW):
		""" Append new weights """
		self.W = newW

	def setBias(self, newBh):
		self.bh = newBh

	def encoder(self, value):
		pass


	def decoder(self, value):
		pass

	def _cost(self, value):
		pass

	def _get_grads(self, cost):
		self.grads = T.grad(cost, self.params)
		return self.grads

	def train(self, batch_size=50, momentum=0.9, weight_decay=0.9):
		pass

	def result(self):
		return self.W



class Loss:
	def __init__(self):
		pass

	def CrossEntropy(self,x, y):
		return -T.sum(x * T.log(y) + (1 - x) * T.log(1 - y), axis=1)
		#return T.sum(T.dot(1 - y, T.log(1 - x)) + T.dot(y, T.log(x)))

	def LSE(self, x, y):
		return T.sum(T.sqrt((x - y)**2))

	def Hinge(self, x):
		return T.max(0, 1 + x)

	def ReLU(self, x):
		return T.max(0, x)

	def ZeroOne(self, x, y):
		pass


class SoftmaxRegression:
	""" Basic implementation of softmax regression(for Stacked Autoencoder)
		http://ufldl.stanford.edu/wiki/index.php/Softmax_Regression

		http://blog.datumbox.com/machine-learning-tutorial-the-multinomial-logistic-regression-softmax-regression/
	"""

	def __init__(self, inp, y, hid_num=30, num_classes=3, numpy_rng=None, theano_inp=None, theano_labels=None, lrate=0.1, inp_num=64):
		self.inp = inp
		self.labels = y
		self.num_classes = num_classes
		self.size = inp.shape[0]
		if theano_inp == None:
			self.x = T.matrix('x')
		else:
			self.x = theano_inp
		self.y = theano_labels
		if theano_labels == None:
			self.y = T.ivector('y')
		if numpy_rng == None:
			self.rng =np.random.RandomState(55)
		else:
			self.rng = numpy_rng
		par = ParametersInit(self.rng, -0.0001, 0.0001)
		self.W = par.get_weights(((inp_num, hid_num)))
		self.bh = theano.shared(np.asarray(np.zeros(hid_num), dtype=theano.config.floatX), name='bh')
		self.lrate = lrate
		self.params = [self.W, self.bh]

	def _get_grads(self, cost):
		return T.grad(cost, self.params)

	def updateParams(self, newparams):
		for i in range(len(newparams)):
			self.params[i] = newparams[i]

	def cost(self, weight_decay=0.9):
		value = T.nnet.softmax(T.dot(self.x, self.W) + self.bh)
		prediction = T.argmax(value, axis=1)
		#result = Loss().CrossEntropy(value, self.y) + T.sum((self.W)**2) * 1/(2 * self.num_classes)
		#result = -T.mean(T.log(value - self.y))
		result = -T.mean(T.log(value)[T.arange(self.y.shape[0]), self.y])
		grads = self._get_grads(result)
		return result, [(param, param - self.lrate * gparam) for param, gparam in zip(self.params, grads)]

	def train(self):
		cost_value, updates = self._cost()
		#func = theano.function([], cost, updates=updates, givens={self.x: self.inp, self.y: self.labels})
		func = theano.function([], cost_value, updates=updates, givens={self.x: self.inp, self.y: self.labels})
		for i in range(100):
			print("iter {0}".format(i))
			print(func())


class Training:
	'''
		Puppet class for trainers
	'''
	def __init__(self):
		pass


class Momentum:
	'''
		Nesterov momentum
	'''
	def __init__(self):
		pass

class ParametersInit:
	def __init__(self, rng, low, high):
		"""
			rng - theano or numpy
			low - low value for init
			high - high value for init
		"""
		self.rng = rng
		self.low = low
		self.high = high

	def get_weights(self, size, name='W'):
		"""
			size in tuple format
			name - current name for weights
		"""
		return self._initW(size, name)

	def _initW(self, size, name):
		return theano.shared(value = \
			np.asarray(
				self.rng.uniform(low=self.low, high=self.high, size=size), dtype=theano.config.floatX
				), name='W')

	def _initW2(self, size, nin, nout, name):
		return theano.shared(value = \
			np.asarray(
				self.rng.uniform(low=-np.sqrt(6)/np.sqrt(nin + nout), high=np.sqrt(6)/np.sqrt(nin + nout), \
					size=size), dtype=theano.config.floatX
				), name='W')


class Autoencoder(AutoencoderPuppet):
	def __init__(self, x=None, theano_input = None, num_vis=100, num_hid=50, numpy_rng=None, lrate=0.001, corruption_level=0, \
		encoder_func='sigmoid', decoder_func = None):
		AutoencoderPuppet.__init__(self, x=x, theano_input=theano_input, num_vis=num_vis, numpy_rng=numpy_rng, corruption_level=corruption_level,\
			encoder_func=encoder_func, decoder_func=decoder_func)

	def append_event(self, iter, event):
		""" Append some event on iteration 
			For example, on iteration 50, change function of activation
		"""
		pass
	def add_theano_input(self, x):
		self.x = x
	def appendInput(self, x):
		self.training = x

	def _forward(self):
		hidden = self.encoder(self.x, self.W)
		#Transpose version of W
		decode = self.decoder(hidden, self.W)
		L = T.mean(Loss().CrossEntropy(self.x, decode))
		#L = T.mean(T.log(decode))
		cost = T.mean(L)
		return cost

	def encoder(self, value, W):
		result = value
		if self.corruption_level > 0:
			result = self._corrupt(value)
		return self._encoder_func(T.dot(result, W) + self.bh)

	def decoder(self, value, W):
		""" Note: In this type of decoder is looks same as encoder """
		result = T.dot(value, W.T) + self.bv
		if self._decoder_func == None:
			return result
		else:
			return self._decoder_func(result)

	def _cost(self):
		cost = self._forward()
		grads = self._get_grads(cost)
		return cost, [(param, param - self.lrate * gparam) for param, gparam in zip(self.params, grads)]

	def output(self):
		return self._forward()

	def updateParams(self, cost):
		""" One step for update params """
		params = [self.W, self.bv, self.bh]
		grads = T.grad(cost, params)
		for i in range(len(self.params)):
			self.params[i] = params[i] - self.lrate * grads[i]

	def gradients(self):
		return self.grads

	def _corrupt(self, value):
		""" Corrupt in case, if corruption_level > 0
		"""
		binom = theano.shared(np.random.binomial(1, self.corruption_level, self.num_vis))
		return value * binom

	def minitoring(self):
		'''
			Monitoring the network training
		'''
		pass

	def train(self, batch_size=0.2, encoder_func='sigmoid', decoder_func = None, iters=100, lrate=0.001):
		""" 
			batch_size in percent, how many data choice from dataset on each iteration
		"""
		'''self._encoder_func = T.nnet.ultra_fast_sigmoid
		self._decoder_func = T.nnet.sigmoid'''
		index = T.lscalar()
		costvalue, updates= self._cost()
		#trainda = theano.function([self.x], costvalue)
		#mini-batch training
		trainda = theano.function([], costvalue, updates=updates, givens={self.x: self.training})
		allerrors = 0
		print("This is input size: jjj", self.num_vis)
			#givens={x: train_set_x[index * batch_size:(index + 1) * batch_size]})
		for i in range(iters):
			current = trainda()
			allerrors += current
			print("Iteration number {0}. Cost is {1}".format(i, current))

		print("Average error: {0}".format(allerrors/(i+1)))


	def _innerTrain(self, data):
		pass


	def predict(self, newvalue):
		""" Predict new data after training """
		pass

def test_AE():
	datasets = load_dataset(dataset)
	batches = train_set_x.get_value(borrow=True).shape[0]/batch_size
	index = T.iscalar()
	x = T.matrix('x')


class HiddenLayer:
	""" 
		Hidden layer for the stacked arch 
	"""
	def __init__(self, theano_inp, num_vis, num_hid, inp=None, activation='sigmoid', numpy_rng=None):
		self.x = theano_inp
		if inp != None:
			""" Numpy input or just array as input """
			self.inp = inp
		self.activation = activation
		if numpy_rng == None:
			numpy_rng = np.random.RandomState()
		par = ParametersInit(numpy_rng, -0.001, 0.001)
		self.W = par.get_weights(((num_vis, num_hid)))
		#Bias init as zero
		self.bh = theano.shared(np.asarray(np.zeros(num_hid), dtype=theano.config.floatX), name='bh')

	def output(self):
		func = T.nnet.sigmoid
		if self.activation == 'tanh':
			func = T.tanh
		return func(T.dot(self.x, self.W) + self.bh)

class StackedAutoencoder(AutoencoderPuppet):
	""" 
	http://www.jmlr.org/papers/volume11/vincent10a/vincent10a.pdf

	"""
	def __init__(self, x, y, num_layers=None, layers=None, corruption_level=0.5, hidlayers=None):
		"""
			|corruption_level| = layers
			pretrain=True - pretraing autoencoder, before main training
			hidlayers - Just a list of hidden neurons on each layer. For example [1024,256,32,10]
		"""

		if hidlayers != None:
			layers = self._costructByNumHidden(hidlayers)
		if num_layers != None:
			self.num_layers = num_layers
			self.layers = []
		elif layers != None:
			self.layers = layers
			self.num_layers = len(layers)
		else:
			self.num_layers = 3
		AutoencoderPuppet.__init__(self, x)
		self.x = T.matrix('x')
		self.y = y
		self.training = x
		self.labels = T.matrix('y')
		self.corruption_level = corruption_level
		#Params from all layers
		self.params = []

	def _innerTrain(self, iters, pretrain):
		if len(self.layers) == 0:
			''' Just construct by default '''
			new_l = self._construct()

		if pretrain == 'glw':
			hidden_layers, new_layers = self._getAEs(self.layers)
			after_pre_train_layers = self._pre_train(new_layers)
			after_fine_tune = self.finetune(self.layers, after_pre_train_layers, self.y)
			return after_fine_tune

	def train(self, iters=5000, pretrain=None):
		"""
			layers - number of DA
			pretrain - glw (Greedy-Layer Wise)
		"""
		if pretrain != None:
			"Return pretrainined params for each layers"
			pretraining = self._innerTrain(iters, pretrain)
			func = theano.function([self.x,self.labels], pretraining)
			func(self.training, self.y)
		#Main training phase
		'''fun1 = self.layers[0]
		fun1.train()
		inp = fun1.output()
		for i in range(1, len(self.layers)):
			fun2 = self.layers[i]
			self.layers[i].add_theano_input(inp.T)
			fun2.train()
			inp = self.layers[i].output().T'''
		'''for layer in new_layers:
			layer.train(batch_size=0.3, encoder_func='tanh', decoder_func='sigmoid')'''


	def targetProp(self, layers):
		""" Target Propagation for StackedAutoencoder. Page 18, How to Credit Assignment...
			layers - is list of AE
		"""
		state = RandomState()
		h0 = state.binomial(self.x.shape[0], p=0.5)
		for i in range(1, self.num_layers):
			layers[i].hidden = T.nnet.sigmoid(layers[i].hidden)
			layers[i-1].corrupted = T.nnet.sigmoid(layers[i-1].corrupt(layers[i-1].hidden))
			layers[i].target = layers[i-1].corrupted
		""" TODO """

	def _construct(self):
		'''
			Construct list of autoencoders (Puppet class need to be is AutoencoderPuppet)
		'''
		return [Autoencoder(None) for i in range(self.num_layers)]

	def _costructByNumHidden(self, listofhidden):
		'''
			Construct autoencoders by num of hidden layers
		'''
		return [Autoencoder(num_hid=i) for i in range(listofhidden)]


	def _getAEs(self, layers):
		""" 
			Construct layers (using shared weights)
			papers:
			Greedy Layer-Wise Training of Deep Networks
			http://papers.nips.cc/paper/3048-greedy-layer-wise-training-of-deep-networks.pdf

			Learning Deep Architectures for AI

			http://www.iro.umontreal.ca/~lisa/publications2/index.php/attachments/single/20 (page 2)

		"""
		numlayers = len(layers)
		hidden_layers = []
		for i in range(0, numlayers):
			""" first(i = 0) input from hidden layer use as from visible"""
			if i == 0:
				layer_input = self.x
				layer_num_vis = self.layers[0].num_vis
				layer_num_hid = self.layers[0].num_hid
			if i != 0:
				layer_input = layers[i-1].x
				layer_num_vis = layers[i-1].num_vis
				layer_num_hid = layers[i-1].num_hid
			hidden_layer = HiddenLayer(layer_input, layer_num_vis, layer_num_hid)
			hidden_layers.append(hidden_layer)
			layers[i].setWeights(hidden_layer.W)
			layers[i].setBias(hidden_layer.bh)

			'''autoencoder_res = layers[i]
			new_layers.append(autoencoder_res)'''
		return hidden_layers, layers


	def _pre_train(self, layers, epoches=1):
		num_layers = len(layers)
		if layers == None or num_layers == 0:
			raise Exception("Length or layers equal to zero or is None")
		pre_trainied=[]

		it = 0
		print("Start pretrain phase on {0} layers".format(num_layers))
		for i in range(len(layers)):
			while it < epoches:
				#Iterate over all inputs
				result_layer = layers[i].output()
				layers[i].updateParams(result_layer)
					#self._updateParams(num_layers)
				it += 1
				#Update params for i-th layer
		print("End of pretrain phase", len(pre_trainied))
		return layers

	def finetune(self, layers, pre_trainedlayers, labels, epoch=10):
		"""
			Forward and backward propagations и градиенты
			layers - original layers , ...

			Finetuning, by definition, is done by using supervised labels to update the weights of a neural network. 
			The goal of finetuning is often not to get good looking features,
			but rather good predictive performance on some classification task.
			http://www.quora.com/Deep-learning-UFLDL-Stack-Autoencoder-exercise-How-to-finetune-without-any-classifier-Softmax-classifier
		"""
		layer_result = []
		#Forward propagation
		print("Start finetuning phase: ")
		for i in range(epoch):
			#First - input from first hidden layer
			layer_input = layers[0].x
			for l in range(1, len(layers)):
				layers[l].add_theano_input(layer_input)
				layer_input =layers[l].output()

			softmax = SoftmaxRegression(layer_input, labels)
			result = softmax.cost()
			softmax.updateParams(result[1])

		#In the last, train logistic regression(Softmax regression) layer
		'''for i in range(1, len(layers)):
			layers[i].add_theano_input(layer_input)
			layer_input =layers[i].output()
		#Append new layer
		#layers[i + 1] = SoftmaxRegression(layers[i].x, self.y)
		for i in range(len(layers)-1,-1,-1):
			grads = layers[i].gradients()
		return layer_result'''

	def predict(self, inp):
		""" Prediction of some value """
		for layer in range(len(self.layers)):
			sigm = self.sigmoid_layer[i]
			inp = sigm.output(input=inp)

class SparseAutoencoder(AutoencoderPuppet):
	def __init__(self, x, num_vis, num_hid, weights=None, bias=None):
		self.inp = x
		self.all_numbers = x.shape[0]
		self.x = T.matrix('x')
		self.W = weights
		if weights == None:
			par = ParametersInit(numpy_rng, 0.0, 1.1)
			self.W = par.get_weights(((num_vis, num_hid)))
		self.bh = theano.shared(np.asarray(np.zeros(num_hid), dtype=theano.config.floatX), name='bh')
		self.bv = theano.shared(np.asarray(np.zeros(num_vis), dtype=theano.config.floatX), name='bv')

	def _KL(self, first, second):
		return T.sum(first * T.log(first/second) + (1 - first) * T.log((1 - first)/(1 - second)))

	def _cost(self, sigma=0.01, beta=None):
		hidden = self.encoder(value, self.W)
		#Transpose version of W
		decode = self.encoder(hiddenself.W.T)
		L = -T.sum(hidden * T.log(decode) + (1 - hidden) * T.log(1 - decode))
		L += self._KL(self.x, decode)
		L +=  sigma * T.sum(W) **2

		#Sparsity cost
		L_sparsity = 1/self.all_numbers * T.sum(hidden)

		grad = T.grad(L, self.params)
		return T.mean(L)



#http://www.cor-lab.de/non-negative-sparse-autoencoder
#http://bigml.cs.tsinghua.edu.cn/~jun/pub/lstm-parallel.pdf

class NNSAE:
	def __init__(self, num_vis, num_hid):
		self.num_vis = num_vis
		self.num_hid = num_hid
		self.W = theano.shared(name='W', data = \
			numpy.asarray(stream.uniform(low=0.0, high=1.0, size=((num_vis,num_hidd)), dtype=theano.config.floatX)))

		self.b1 = theano.shared(name='W', data = \
			numpy.asarray(stream.uniform(low=0.0, high=1.0, size=num_hid, dtype=theano.config.floatX)))

		self.b1 = theano.shared(name='W', data = \
			numpy.asarray(stream.uniform(low=0.0, high=1.0, size=num_hid, dtype=theano.config.floatX)))

	def train(self):
		lrate = 0.9

		self.W = self.W + lrate * err


class StackedConvolutionalAutoencoder(AutoencoderPuppet):
	"""
		http://people.idsia.ch/~ciresan/data/icann2011.pdf
	"""
	def __init__(self, x):
		AutoencoderPuppet.__init__(self, x)


class GAE:
	"""
	Generalized DAE
	http://papers.nips.cc/paper/5023-generalized-denoising-auto-encoders-as-generative-models.pdf

	На входе, используется список из AE.
	На выходе - сгенерированные новые примеры
	"""
	def __init__(self, x, num_hidd, aelayers, layers=None, rng=None, theano_rng=None):
		'''
			aelayers - list of autoencoders
			layers - list of number of hidden units in autoencoders
		'''
		self.x = x
		self.rng = rng
		self.theano_rng = theano_rng
		self.num_hid = numpy_rng
		self.layers = layers
		if rng == None:
			self.rng = np.random.RandomState()
		if theano_rng == None:
			self.theano_rng = RandomStreams()
		par = ParametersInit(numpy_rng, 0.0, 1.1)
		self.W = par.get_weights(((num_vis, num_hid)))
		self.params = [self.W, self.bias]


	def _corrupt(self, value):
		return self.rng.binomial(value.shape[0], 0.5)

	def _get_grads(self, cost):
		return T.grad(cost, self.params)

	def _get_cost(self, inp):
		"""Sample from training examples x """
		sample_x_idx = self.rng.binomial(self.num_hid, propb)
		sample_x = T.dot(self.x, sample_x_idx)
		hidden = T.dot(sample_x, self.W) + self.b
		reconstruct = T.dot(hidden, self.W.T)
		""" Corrupt this sample """
		corrupted_sample_x = self._corrupt(sample_x)
		""" Sample corrupted input """
		sample_corr = self.rng.binomial(self.num_hid, propb)
		sample_corrupted = T.dot(corrupted_sample_x, sample_corr)
		#Добавить этот сэмпл в скрытый слой (Приблизительная реализация)
		self.x[self.numvis + 1] = sample_corrupted
		#Cost need to be with negative log-likelihood
		cost = -T.mean(T.log(self.params))
		grads = self._get_grads(cost)
		return cost, [(para, param - lrate * gparam) for (para, gparam) in zip(self.params, grads)]

	def walkback_step(self, prob=None,p=0.5):
		""" One step for walkback process """
		result = []
		''' p - default probability '''
		if prob != None:
			''' Sample from training examples X '''
			samplx_x_idx = self.rng.binomial(self.num_hid, prob)
		''' In the case if prob = None, pick just one example from training set '''

		samplx_x_idx = self.x[np.random.randint(low=0, high=self.num_vis)]
		#Corrupt
		r = self.rng.binomial(self.numvis, p)
		samplx_x_corruptd = T.dot(self.x, r)
		u = np.random.random()
		if u > p:
			result.append(samplx_x_corruptd)
		#Append (sample_x, sample_x_corrupted) as additional training example
		#In the last step, sample from P(X|\hat X*)

	def _update_layers():
		for layer in self.layers:
			hidden = T.nnet.sigmoid(layer.x, self.W)
			#Corrupt input


	def train(self, propb):
		"""
			Main training phase
		"""
		pass


class MarginalizedDenoisingAutoencoder(AutoencoderPuppet):
	"""
		http://arxiv.org/pdf/1206.4683v1.pdf
	"""
	def __init__(self, x, num_vis, num_hid):
		AutoencoderPuppet.__init__(self, x)
		par = ParametersInit(numpy_rng, 0.0, 1.1)
		self.W = par.get_weights(((num_vis, num_hid)))
		self.params = [self.W]

	def _appendBias(value):
		newvalue = T.matrix('xb')

	def _cost(self, p, activation='sigmoid'):
		if activation == 'sigmoid':
			pass
		self.x = self._appendBias(self.x)
		value = np.ones((d, 1)) * (1 - p)
		value[d-1] = 1
		corrm = T.shared(np.dot(value, value.T))
		prod = T.dot(self.x, self.x)
		Q = T.dot(prod, corrm)
		R = np.dot(value, T.diag(prod))
		rep = T.dot(prod, np.repeat(value, d))
		Wp = T.dot(self.W, self.W)
		hidden = T.tanh(np.dot(self.x, W))
		grads = self._get_grads(Loss().CrossEntropy(hidden, decode))

	def _get_grads(self, cost):
		return T.grad(self._cost, self.params)

	def train(self, p, iters=5000):
		for i in range(iters):
			cost = self._cost(p)
			grads = self._get_grads(cost)


def test_MarginalizedDAE():
	x = np.array([5,4,8,5,6,9,6,3,2,1,4,8,5,8,9])
	mda = MarginalizedDenoisingAutoencoder(x)
	mda.train(0.35)
	#Predict new values
	mda.predict()


class MADE(AutoencoderPuppet):
	''' Masked Autoencoder for Distribution Estimation '''

	def __init__(self, num_vis, num_hid, x=None, theano_input = None, learning_rate=0.01, numpy_rng=None):
		if numpy_rng == None:
			numpy_rng = np.random.RandomState()
		par = ParametersInit(numpy_rng, 0.0, 1.1)
		self.W = par.get_weights(((num_vis, num_hid)))
		self.V = par.get_weights(((num_vis, num_hid)))
		self.biash = theano.shared(np.asarray(np.zeros(num_hid), dtype=theano.config.floatX), name='bh')
		#self.biasv = theano.shared(np.asarray(np.zeros(num_vis), dtype=theano.config.floatX), name='bv')
		self.x = theano_input
		self.num_vis = num_vis
		self.num_hid = num_hid
		if theano_input == None:
			self.x = T.matrix('x')
		else:
			self.x = theano_input
		self.inp = x
		self.learning_rate = learning_rate
		self.params = [self.W, self.V, self.biash]

	def _getMask(self, num_layers):
		if num_layers == 1:
			maskhidden = theano.shared(np.random.randint(1,self.num_hid - 1,self.num_hid), name='maskh')
			maskvisible = theano.shared(np.random.randint(0,self.num_vis - 1,self.num_vis), name='maskv')
			return maskvisible, maskhidden
		elif num_layers > 1:
			maskhidden = []
			maskvisible = []
			for layer in range(num_layers):
				maskvisible.append(T.shared(np.random.randint(1,self.num_hid - 1,self.num_hid), name='maskh'))
				maskhidden.append(T.shared(np.random.randint(0,self.num_vis - 1,self.num_vis), name='maskv'))
			return maskvisible, maskhidden

	def _getBinaryMask(self, layers, num_layers):
		''' 1 if mask[layer1[i]] > mask[layer0[i]] otherwise 0 
			input - layers of mask, previusly generated from _getMask
		'''
		binary = T.matrix('bin')
		for layer in range(1, num_layers+1):
			for i in range(layers[i].shape[0]):
				for j in range(layers[i].shape[1]):
					pass

	def _cost(self):
		#Sample m^l vectors
		#get masks
		#MAskV, MaskH - должны быть бинарные матрицы
		MaskV, MaskH = self._getMask(1)
		hidden = T.dot(self.x, self.W * MaskH) + self.biash
		corrupt = T.nnet.sigmoid(T.dot(hidden, self.V * MaskV) + self.biash)
		cost = T.exp(T.sum(T.log(self.x * T.log(corrupt) + (1 - self.x) * T.log(1 - corrupt))))

		#Можно добавить Cross-entropy или Negative log likelihood
		grad = T.grad(-T.log(cost), self.params)
		return hidden[0][0] #, [(param, param - self.learning_rate * gparam) for (param, gparam) in zip(self.params, grad)]

	def _compute_gradients(self, corrupt, hidden):
		""" Compute gradients step-by-step """
		tmp = corrupt - self.x
		c_new = tmp
		V_new = T.dot(tmp, hidden) * self.V
		tmp = T.dot(tmp.T, self.V * self.W)
		#Compute with all layers
		

	def train(self):
		""" Main function of training """
		cost = self._cost()
		func = theano.function([self.x], cost)
		print(func(self.inp))
		'''func = theano.function([], cost, updates=updates, givens={self.x: self.inp})
		for i in range(10):
			print("cost: ", func())'''

class ContrastiveAutoEncoder(Autoencoder):
	""" Basic contrastive AE """
	def __init__(self, x, theano_input = None, num_vis=100, num_hid=50, numpy_rng=None, lrate=0.001, corruption_level=0, \
		encoder_func='sigmoid', decoder_func = None):
		AutoencoderPuppet.__init__(self,x=x, theano_input=theano_input, num_vis=num_vis, numpy_rng=numpy_rng, corruption_level=corruption_level,\
			encoder_func=encoder_func, decoder_func=decoder_func)
		self.lmb = lrate

	def jacobian(self, hidden, W):
		''' trick from deeplearning.net tutorial '''
		return T.reshape(hidden * (1 - hidden), (self.num_vis, 1, self.num_hid)) * T.reshape(W, (1, self.num_vis, self.num_hid))

	def _cost(self, con_level=0.05):
		hidden = self.encoder(self.x, self.W)
		#Transpose version of W
		decode = self.decoder(hidden, self.W)
		J = T.sum(self.jacobian(hidden, self.W)**2)/self.num_vis
		L = T.mean(Loss().CrossEntropy(self.x, decode)) + con_level * T.mean(J)
		#L = T.sum(decode)
		grads = self._get_grads(L)
		return L, [(param, param - self.lrate * gradparam) for (param, gradparam) in zip(self.params, grads)]

	def train(self):
		cost, updates= self._cost()
		func = theano.function([], cost, updates=updates, givens={self.x: self.training})
		for i in range(5):
			print(func())


class BlockedAutoencoder:
	''' Кодируем и декодируем поблочно
		Разделяем входные юниты на n блоков, кодируем их(с разными степенями повреждения, замем, декодируем обратно)
	 '''
	def __init__(self, x=None, theano_input = None, num_vis=100, num_hid=50, numpy_rng=None, lrate=0.001, corruption_level=0, \
		encoder_func='sigmoid', decoder_func = None):
		AutoencoderPuppet.__init__(self,x=x, theano_input=theano_input, num_vis=num_vis, numpy_rng=numpy_rng, corruption_level=corruption_level,\
			encoder_func=encoder_func, decoder_func=decoder_func, block_num=5)
		self.lmb = lrate
		self.block_num = block_num

	def _encoder(self, value, weights):
		block = self.num_vis/block_num
		blocks = []
		for i in range(self.block_num):
			result = value[i:block]
			blocks.append(result)
			i += block


class Model:
	""" Basic building block for running autoencoders, howewer
		classes for autoencoders can be use without this class
	 """
	def __init__(self, *args, **kwargs):
		self.inp = kwargs.get('input')
		self.dataset = None
		if self.inp == None:
			raise Exception("Can't find input for learning")

	def addDataset(self, path=None, title=None):
		if title == 'mnist' and path == None:
			self.dataset = get_mnist()
		if title == 'mnist' and path != None:
			self.dataset = load_mnist(path)

	def add(self, title, aemodel):
		if not isinstance(aemodel, AutoencoderPuppet):
			raise Exception("This model is not from AutoencoderPuppet class")

	def addCost(self, title, cost):
		""" Append cost function for ae model 
			title - name of ae, which was append
		"""
		pass
	def addTrainingAlgorithm(self, title='SGD'):
		if title == 'SGD':
			print("This is SGD")
		if title == 'BGD':
			print("This is BGD", ...)

	def train(self, batchtype, batchsize):
		""" Start trainin model 
			batchtype - (minibatch, online batch)
		"""
		pass


def load_mnist(path):
	"""
		Load mnist for python 3
	"""
	import gzip
	f = gzip.open(path, 'rb')
	return pickle.load(f)

def get_mnist():
	"""
		Download mnist from ...
	"""
	pass