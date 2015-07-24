import numpy as np
import theano
import theano.tensor as T
from theano.tensor.shared_randomstreams import RandomStreams
import theano.sandbox.rng_mrg as RNG_MRG
from theano.printing import pprint
import pickle
from softmaxregression import SoftmaxRegression
from parametersinit import ParametersInit


class AutoencoderPuppet:

    """ Basic build class for any AE """

    def __init__(self, x=None, theano_input=None, num_vis=100, num_hid=50, numpy_rng=None, theano_rng=None, lrate=0.001, corruption_level=0,
                 encoder_func='sigmoid', decoder_func=None, momentum=0.9, tied_weights=True, cost_func='ce'):
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
            momentum - for training to help out from local minima (TODO: Nesterov momentum)
            cost_func - cost function(ce, lse)
            (bouth can be None)
        '''
        self.training = x
        self.namedparams = {}
        if theano_input == None:
            self.x = T.matrix('x')
        else:
            self.x = theano_input
        self.x = T.matrix('x')
        self.numpy_rng = numpy_rng
        if numpy_rng == None:
            self.numpy_rng = np.random.RandomState()
        self.theano_rng = theano_rng
        if theano_rng == None:
            self.theano_rng = RandomStreams()
        w_init = 4 * np.sqrt(6. / (num_hid + num_vis))
        par = ParametersInit(self.numpy_rng, -w_init, w_init)
        self.par = par
        self.W = par.get_weights((num_vis, num_hid), 'W')
        if tied_weights == False:
            self.W2 = par.get_weights((num_hid, num_vis), 'W2')
        # Bias init as zero
        self.bh = theano.shared(
            np.asarray(np.zeros(num_hid), dtype=theano.config.floatX), name='bh')
        self.bv = theano.shared(
            np.asarray(np.zeros(num_vis), dtype=theano.config.floatX), name='bv')
        self.params = [self.W, self.bv, self.bh]
        self.lrate = lrate
        self.corruption_level = corruption_level
        self.num_vis = num_vis
        self.num_hid = num_hid
        self.momentum = momentum
        self.cost_func = cost_func
        self._set_encoder_func(encoder_func, decoder_func)
        self.params = []

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
            self._decoder_func = T.nnet.softmax

    '''def updateParams(self, cost):
        """ One step for update params
        """
        grads = self._get_grads(cost)'''

    def initWeights(self, weight, heights, name):
        """ Init weights needs for more easy initialization
            of weights for complex models
        """
        self.namedparams[name] = self.par.get_weights((weight, heights), name)


    def setWeights(self, newW):
        """ Append new weights """
        self.W = newW

    def setBias(self, newBh):
        self.bh = newBh

    def encoder(self, value):
        pass

    def decoder(self, value):
        pass

    def _forward(self):
        """ Result after forward propagation.
            Usual, return two values - cost and updates
        """
        pass

    def _cost(self, value):
        pass

    def _get_grads(self, cost):
        self.grads = T.grad(cost, self.params)
        return self.grads

    def train(self, batch_size=50, momentum=0.9, weight_decay=0.9):
        pass

    def score(self, x1, x2):
        return T.mean(T.neq(x1, x2))

    def test_model(self, validation):
        """ Validation set (typical 0.1 part of training set)
            Run this after training model and return mean error
        """
        pass

    def result(self):
        return self.W

    def saveParams(self, path1, path2):
        """ Save weights to path
        """
        np.save(path, self.W.get_value())
        np.save(path2, self.b.get_value())

    def loadWeights(self, path):
        self.W = np.load(path, self.W.get_value())


class Loss:

    def __init__(self):
        pass

    def CrossEntropy(self, x, y):
        return -T.sum(x * T.log(y) + (1 - x) * T.log(1 - y), axis=1)
        # return -T.sum(x * T.log(y) + (1 - x) * T.log(1 - y), axis=1)

    def LSE(self, x, y):
        return T.sum(T.sqrt((x - y)**2))

    def Hinge(self, x):
        return T.max(0, 1 + x)

    def ReLU(self, x):
        return T.max(0, x)

    def ZeroOne(self, x, y):
        pass

class Training:

    '''
        Puppet class for training
    '''

    def __init__(self, learning_rate,  momentum=0.99):
        self.learning_rate = learning_rate
        self.momentum = momentum

    def addVelocityShape(self, shape):
        self.velocity = T.zeros(shape)

    def sgd(self, oldparam, newparam):
        vel_new = oldparam * self.momentum - self.learning_rate * newparam
        self.velocity = vel_new
        # return oldparam + vel_new
        return oldparam - newparam * self.learning_rate


class StepRule:

    def __init__(self, initalpha, args, kwargs):
        self.initalpha = initalpha

    """ Step rule for gradient descent """

    def step(self):
        raise NotImplementError()


class StepRule1(StepRule):

    def __init__(self, initalpha):
        StepRule.__init__(initalpha)

    def step(self):
        pass


class Momentum:

    """ Implementation of momentum for gradient descent """

    def compute(self, oldvalue, newvalue):
        vel = theano.shared(0)
        result = self.momentum * vel + self.steprate * newvalue
        return oldvalue - result


class Autoencoder(AutoencoderPuppet):

    def __init__(self, x=None, theano_input=None, num_vis=100, num_hid=50, numpy_rng=None, lrate=0.001, momentum=0.9, corruption_level=0,
                 encoder_func='sigmoid', decoder_func=None, tied_weights=True):
        AutoencoderPuppet.__init__(self, x=x, theano_input=theano_input, num_vis=num_vis, num_hid=num_hid, numpy_rng=numpy_rng, corruption_level=corruption_level,
                                   encoder_func=encoder_func, decoder_func=decoder_func, momentum=momentum, tied_weights=tied_weights, lrate=lrate)
        self.params = [self.W, self.bh]
        self.traindata = Training(lrate, momentum=momentum)

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
        # Transpose version of W
        decode = self.decoder(hidden, self.W)
        L = T.mean(Loss().CrossEntropy(self.x, decode)) + \
            0.0005 * self._regularization()
        #L = T.mean(T.log(decode))
        return L, hidden

    def encoder(self, value, W):
        result = value
        if self.corruption_level > 0:
            result = self._corrupt(value)
        return self._encoder_func(T.dot(result, W) + self.bh)

    def decoder(self, value, W):
        """ Note: In this type of decoder is looks same as encoder """
        result = T.dot(value, W.T) + self.bv
        if self._decoder_func == None:
            return T.nnet.sigmoid(result)
        else:
            return self._decoder_func(result)

    def _regularization(self):
        ''' Append L1 and L2 regularization to cost function '''
        result = theano.shared(0.0)
        for param in self.params:
            result += T.sum(param**2)
        return result

    def _cost(self):
        cost, hidden = self._forward()
        grads = self._get_grads(cost)
        self.traindata.addVelocityShape(grads[0].shape)
        return cost, [(param, self.traindata.sgd(param, gparam)) for param, gparam in zip(self.params, grads)]

    def output(self):
        cost, hidden = self._forward()
        return cost, hidden

    def updateParams(self, cost):
        params = [self.W, self.bv, self.bh]
        self.momentum = 0.9
        grads = T.grad(cost, params)
        self.traindata.addVelocityShape(grads[0].shape)
        return cost, [(param, param - (self.lrate * gparam)) for param, gparam in zip(self.params, grads)]

    def gradients(self):
        return self.grads

    def _corrupt(self, value):
        """ Corrupt in case, if corruption_level > 0
        """
        binom = theano.shared(
            np.random.binomial(1, self.corruption_level, self.num_vis))
        return value * binom

    def minitoring(self):
        '''
            Monitoring the network training
        '''
        pass

    def train(self, batch_size=10, encoder_func='sigmoid', decoder_func=None, iters=100, lrate=0.001):
        """
            batch_size in percent, how many data choice from dataset on each iteration
        """
        '''self._encoder_func = T.nnet.ultra_fast_sigmoid
        self._decoder_func = T.nnet.sigmoid'''

        # mini-batch index
        #index = T.lscalar()
        index = 0
        costvalue, updates = self._cost()
        #trainda = theano.function([self.x], costvalue)

        train_batches = self.training.shape[0] / batch_size
        #This is bug
        inpdata = self.training[index * batch_size: (index + 1) * batch_size]
        trainda = theano.function([], costvalue, updates=updates,
                                  givens={self.x: inpdata})
        allerrors = 0
        print("Batch size {0}".format(batch_size))
        # givens={x: train_set_x[index * batch_size:(index + 1) * batch_size]})
        result_cost = []
        for i in range(iters):
            current = 0
            for idx in range(int(abs(train_batches))):
                result = trainda()
                current += result
                index = idx
            print(
                "Iteration number {0}. TRaining error is {1}".format(i, current / index))

        print("Average error: {0}".format(allerrors / (i + 1)))

    def test_model(self, validation):
        pass

    def _innerTrain(self, data):
        pass

    def predict(self, newvalue):
        """ Predict new data after training """
        pass


def test_AE():
    datasets = load_dataset(dataset)
    batches = train_set_x.get_value(borrow=True).shape[0] / batch_size
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
        self.W = par.get_weights(((num_vis, num_hid)), 'W')
        # Bias init as zero
        self.bh = theano.shared(
            np.asarray(np.zeros(num_hid), dtype=theano.config.floatX), name='bh')

    def output(self):
        func = T.nnet.sigmoid
        if self.activation == 'tanh':
            func = T.tanh
        return func(T.dot(self.x, self.W) + self.bh)

class GatedAutoencoder(AutoencoderPuppet):
    ''' paper:
        Generative class-conditional denoising autoencoders
    '''
    def __init__(self,x, y, num_vis, num_hid, num_out, num_fa):
        AutoencoderPuppet.__init__(self, x=x, num_vis=num_vis, num_hid=num_hid)
        self.x = T.matrix('x')
        self.y = T.matrix('y')
        self.inp = x
        self.labels = y
        self.initWeights(num_fa, num_hid, 'Wh')
        self.initWeights(num_out, num_fa, 'Wy')
        self.initWeights(num_vis, num_fa, 'Wx')
        self.bh = theano.shared(
            np.asarray(np.zeros(num_hid), dtype=theano.config.floatX), name='bh')
        self.bx = theano.shared(
            np.asarray(np.zeros(num_vis), dtype=theano.config.floatX), name='bx')
        self.params_names = ['Wh', 'Wy', 'Wx']
        self.params = [self.bh, self.bx]


    def forward(self):
        Wh = self.namedparams['Wh']
        Wx = self.namedparams['Wx']
        Wy = self.namedparams['Wy']
        hidden = T.nnet.sigmoid(T.dot((T.dot(self.x, Wx) * T.dot(self.y, Wy)), Wh))
        return T.tanh(T.dot((T.dot(hidden, Wh.T) * T.dot(self.y, Wy)), Wx.T))

    def cost(self, xorig, xhat):
        return T.nnet.categorical_crossentropy(xorig, xhat)

    def fit(self):
        self.params = list(self.namedparams.values())
        forw = self.forward()
        cost_result = T.mean(self.cost(self.x, forw))
        grad = T.grad(cost_result, self.params)
        return cost_result, [(old, old - 0.001 * newparam) for(old, newparam) in zip(self.params, grad)]

    def train(self, iters=100):
        value, updates = self.fit()
        for i in range(iters):
            func = theano.function([], value, updates=updates, givens={self.x: self.inp, self.y: self.labels})
            print(func())

class StackedAutoencoder:

    """
    http://www.jmlr.org/papers/volume11/vincent10a/vincent10a.pdf

    TODO: Append to minimization cost function by BFGS method
    Deep learning via Stacked Sparse Autoencoders for Automated Voxel-Wise Brain

    """

    def __init__(self, x, y, num_layers=None, layers=None, corruption_level=0.5, hidlayers=None, num_output=10):
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
        self.x = T.matrix('x')
        self.y = T.ivector('y')
        self.training = x
        self.labels = y
        self.corruption_level = corruption_level
        self.num_output = num_output
        # Params from all layers
        self.params = []

    def _innerTrain(self, iters, pretrain):
        if len(self.layers) == 0:
            ''' Just construct by default '''
            new_l = self._construct()

        if pretrain == 'glw':
            hidden_layers, new_layers = self._getAEs(self.layers)
            #after_pre_train_layers = self._pre_train(new_layers)
            after_fine_tune = self.start_finetune(self.layers)
            return after_fine_tune

    def train(self, iters=5000, pretrain=None, method='backprop'):
        """
            layers - number of DA
            pretrain - glw (Greedy-Layer Wise)
        """
        if pretrain != None:
            "Return pretrainined params for each layers"
            pretraining = self._innerTrain(iters, pretrain)

    def targetProp(self, layers):
        """ Target Propagation for StackedAutoencoder. Page 18, How to Credit Assignment...
            layers - is list of AE
            Paper: Difference Target Propagation(backprop free)
            http://arxiv.org/abs/1412.7525
        """
        state = RandomState()
        h0 = state.binomial(self.x.shape[0], p=0.5)
        for i in range(1, self.num_layers):
            layers[i].hidden = T.nnet.sigmoid(layers[i].hidden)
            layers[
                i - 1].corrupted = T.nnet.sigmoid(layers[i - 1].corrupt(layers[i - 1].hidden))
            layers[i].target = layers[i - 1].corrupted
        """ TODO """

    def getAllParams(self, layers):
        params = []
        for layer in layers:
            params.extend(layer.params)
        return params

    def targetProp2(self, hiddens):
        """ hiddens precomputed hidden layers from network """

        num_layers = len(hiddens)
        # First, compute target from top hidden layer
        globalloss = T.grad(self.cost, self.getAllParams(hiddens))
        firsttarget = hidden[num_layers - 1] - globalloss
        # Next, compute targets for lower layers
        for i in range(len(hiddens), 0, 1):
            hiddens[i - 1] = hiddens[i - 1] - \
                invfunc(firsttarget) + invfunc(hiddens[i])

        # Update params for inverse mapping

        # Update params for feedforward mapping

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
                cost, layer_input = layers[i - 1].output()
                layer_num_vis = layers[i - 1].num_vis
                layer_num_hid = layers[i - 1].num_hid
            hidden_layer = HiddenLayer(
                layer_input, layer_num_vis, layer_num_hid)
            hidden_layers.append(hidden_layer)
            layers[i].add_theano_input(layer_input)
            # layers[i].setWeights(hidden_layer.W)
            # layers[i].setBias(hidden_layer.bh)

            '''autoencoder_res = layers[i]
            new_layers.append(autoencoder_res)'''
        return hidden_layers, layers

    def _pre_train(self, layers, epoches=10):
        """ Unsupervised phase """
        num_layers = len(layers)
        if layers == None or num_layers == 0:
            raise Exception("Length or layers equal to zero or is None")
        pre_trainied = []

        print("Start pretrain phase on {0} layers".format(num_layers))
        for i in range(len(layers)):
            print("Start pretrain phase on {0} layer".format(i + 1))
            it = 0
            while it < epoches:
                # Iterate over all inputs
                result_cost, hidden_fun = layers[i].output()
                cost, updates = layers[i].updateParams(result_cost)
                func = theano.function(
                    [], cost, updates=updates, givens={self.x: self.training})
                out_cost = func()
                print(
                    "Layer {0}, iteration {1}, cost {2}".format(i + 1, it, out_cost))

                it += 1
                # Update params for i-th layer
        print("End of pretrain phase", len(layers))
        return layers

    def finetune(self, layers, epoch=10):
        """
            Supervised phase
            Forward and backward propagations и градиенты
            layers - original layers , ...

            Finetuning, by definition, is done by using supervised labels to update the weights of a neural network.
            The goal of finetuning is often not to get good looking features,
            but rather good predictive performance on some classification task.
            http://www.quora.com/Deep-learning-UFLDL-Stack-Autoencoder-exercise-How-to-finetune-without-any-classifier-Softmax-classifier
        """
        # Forward propagation
        # First - input from first hidden layer
        layer_input = self.x
        params = []
        for l in range(len(layers)):
            layers[l].add_theano_input(layer_input)
            layer_cost, newinput = layers[l].output()
            params.extend(layers[l].params)
            layer_input = newinput
        '''for layer in layers:
            """ Get params from all layers """
            params.extend(layer.params)'''

        # Set output layer on the top of the network
        softmax = SoftmaxRegression(theano_labels=self.y, inp_num=15, hid_num=self.num_output)
        softmax.add_theano_input(layer_input)
        cost= softmax.cost()
        params.extend(softmax.params)
        grads = T.grad(T.mean(cost), params)
        return T.mean(cost), [(oldparam, oldparam - 0.001 * newparam) for
         (oldparam, newparam) in zip(params, grads)]
        #return cost

    def start_finetune(self, layers, epoch=10):
        print("Start finetuning phase: ")
        cost, updates = self.finetune(layers)
        value = theano.function([],
                cost, updates=updates, givens={self.x: self.training, self.y: self.labels})
        print(value())
        print("End of finetune phase")
        '''cost, updates = self.finetune(layers, labels)
        func = theano.function([], cost, updates=updates, givens={self.x: self.training})
        for i in range(epoch):
            result_cost = func()
            print("Cost: {0}".format(result_cost))
        print("End of finetuning phase")'''

        # In the last, train logistic regression(Softmax regression) layer
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
        AutoencoderPuppet.__init__(self, x=x, num_vis=num_vis, num_hid=num_hid)
        self.inp = x
        self.nhid = num_hid
        self.all_numbers = x.shape[0]
        self.x = T.matrix('x')
        self.initWeights(num_vis, num_hid, 'Wh')
        self.bh = theano.shared(
            np.asarray(np.zeros(num_hid), dtype=theano.config.floatX), name='bh')
        self.bv = theano.shared(
            np.asarray(np.zeros(num_vis), dtype=theano.config.floatX), name='bv')

    def _KL(self, first, second):
        return T.sum(((first * T.log(first)) - first * T.log(second)) + ((1 - first) * T.log(1 - first)) - ((1 - first) * T.log(1 - second)))

    def _penalty(self, hidden, sigma, regularization_level=1e-4):
        hidden_level = T.extra_ops.repeat(sigma, self.nhid)
        avg = T.mean(hidden, axis=0)
        after_kl = self._KL(hidden_level, avg)
        return regularization_level * after_kl

    def encoder(self, x, W, b):
        return T.nnet.sigmoid(T.dot(x, W) + b)

    def _cost(self, sigma=0.01, beta=None):
        W = self.namedparams['Wh']
        hidden = self.encoder(self.x, W, self.bh)
        # Transpose version of W
        decode = self.encoder(hidden, W.T, self.bv)
        L = T.sum(T.nnet.categorical_crossentropy(self.x, decode))
        L += self._penalty(hidden, sigma)
        params = []
        params.extend(list(self.namedparams.values()))
        params.append(self.bh)
        params.append(self.bv)
        grad = T.grad(T.mean(L), params)
        return T.mean(L), [(old, old - 0.001 * newparam) for (old, newparam) in zip(params, grad)]

    def train(self, iters=100):
        cost, updates = self._cost()
        for i in range(iters):
            func = theano.function([], cost, updates=updates, givens = {self.x:self.inp})
            print(func())



class KSparseAutoencoder(AutoencoderPuppet):
    ''' Implementation of k-sparse autoencoder
    '''
    def __init__(self, x, num_vis, num_hid, weights=None, bias=None):
        AutoencoderPuppet.__init__(self, x=x, num_vis=num_vis, num_hid=num_hid)
        self.inp = x
        self.nhid = num_hid
        self.all_numbers = x.shape[0]
        self.x = T.matrix('x')
        self.initWeights(num_vis, num_hid, 'Wh')
        self.bh = theano.shared(
            np.asarray(np.zeros(num_hid), dtype=theano.config.floatX), name='bh')
        self.bv = theano.shared(
            np.asarray(np.zeros(num_vis), dtype=theano.config.floatX), name='bv')

    def encode(self, xn, W, b):
        return T.nnet.sigmoid(T.dot(xn, W) + b)

    def _k_highest(self, k, value):
        return value

    def _cost(self, k):
        W = self.namedparams['Wh']
        hidden = self.encode(self.x, W)
        after = self._k_highest(k, hidden)



class TwoCostAutoencoder(AutoencoderPuppet):

    """ Using both supervised and unsupervised cost
        http://arxiv.org/pdf/1412.6583v3.pdf
    """

    def __init__(self, x=None, y=None, theano_input=None, num_vis=100, num_hid=50, numpy_rng=None, lrate=0.001,
                 momentum=0.9, corruption_level=0, encoder_func='sigmoid', decoder_func=None):
        AutoencoderPuppet.__init__(self, x=x, theano_input=theano_input, num_vis=num_vis, numpy_rng=numpy_rng, corruption_level=corruption_level,
                                    encoder_func=encoder_func, decoder_func=decoder_func, momentum=momentum, tied_weights=True, cost_func='lse')
        self.x = T.matrix('x')
        self.y = T.matrix('y')

        def _cost(self, beta=0.01, sigma=0.05):
            #hidden = self.encoder(self.x)
            hidden = self.encoder(self.x, corrupt=self.corruption_level)
            decode = self.encoder(self.x, W.T)
            yhat = T.nnet.softmax(T.dot(hidden2, Wy) + self.by)
            cost1 = T.sum(T.sqr(self.x, decode))
            cost2 = T.sum(self.y * T.log(yhat))
            cost3 = sigma * C
            cost = cost1 + beta * cost2 + cost3



# http://www.cor-lab.de/non-negative-sparse-autoencoder
# http://bigml.cs.tsinghua.edu.cn/~jun/pub/lstm-parallel.pdf

class NNSAE(AutoencoderPuppet):

    ''' Use non-negative weights tied matrix for training
        TODO: Try to start learning after ReLU
    '''

    def __init__(self, x=None, theano_input=None, num_vis=100, num_hid=50, numpy_rng=None, lrate=0.001, momentum=0.9, corruption_level=0,
                 encoder_func='sigmoid', decoder_func=None):
        NNSAE.__init__(self, x=x, theano_input=theano_input, num_vis=num_vis, numpy_rng=numpy_rng, corruption_level=corruption_level,
                       encoder_func=encoder_func, decoder_func=decoder_func, momentum=momentum, tied_weights=True, cost_func='lse')

        par = ParametersInit(np.random.RandomState(), 0.0001, 0.001)
        self.W = par.get_weights(((num_vis, num_hid), 'W'))

    def encoder(self, value):
        return self._encoder_func(T.dot(value, self.W) + self.bh)

    def decoder(self, value):
        return self._decoder_func(T.dot(value, self.W.T) + self.bv)

    def _forward(self):
        encode = self.encoder(self.x)
        decode = self.decoder(encode)
        cost = self._cost(encode, decode)

    def train(self):
        lrate = 0.9

        self.W = self.W + lrate * err


class StackedConvolutionalAutoencoder(AutoencoderPuppet):

    """
        http://people.idsia.ch/~ciresan/data/icann2011.pdf
    """

    def __init__(self, x):
        AutoencoderPuppet.__init__(self, x)


class Corruption:

    """ Basic class for corruption for any types of autoencoders """

    def __init__(self, theano_rng=None):
        self.theano_rng = theano_rng
        if theano_rng == None:
            self.theano_rng = RandomStreams()

    def binomial(self, data, p, n, size):
        return value * self.theano_rng.binomial(size=(size,), n=n, p=p)

    def gaussian(self, data, avg, std, size):
        return data + self.theano_rng.normal(size=size, avg=0.0, std=0.05)


class GAE:

    """
    Generalized DAE
    http://papers.nips.cc/paper/5023-generalized-denoising-auto-encoders-as-generative-models.pdf

    На входе, используется список из AE.
    На выходе - сгенерированные новые примеры

    Simple one hidden layer(encoder-decoder) implementation
    """

    def __init__(self, x, num_vis, num_hid, aelayers, layers=None, numpy_rng=None, theano_rng=None, walkback_steps=0,
                 momentum=0.9):
        '''
            aelayers - list of autoencoders
            layers - list of number of hidden units in autoencoders
            walkback_steps. If walkback steps is 0, use standard algorithms without walkbacks
        '''
        self.inp = x
        self.x = T.matrix('x')
        self.numpy_rng = numpy_rng
        self.theano_rng = theano_rng
        self.num_vis = num_vis
        self.num_hid = num_hid
        self.layers = layers
        self.walkback_steps = walkback_steps
        self.momentum = momentum
        if numpy_rng == None:
            self.numpy_rng = np.random.RandomState()
        if theano_rng == None:
            self.theano_rng = RandomStreams()
        self.corruption = Corruption(theano_rng=theano_rng)
        par = ParametersInit(self.numpy_rng, -0.001, 0.001)
        self.W = par.get_weights(((num_vis, num_hid)), "W")
        self.bh = theano.shared(
            np.asarray(np.zeros(num_hid), dtype=theano.config.floatX), name='bh')
        self.bv = theano.shared(
            np.asarray(np.zeros(num_vis), dtype=theano.config.floatX), name='bv')
        self.params = [self.W, self.bh, self.bv]

    def _corrupt_binomial(self, value, p=0.5):
        """ Corrupt with binomial distribution """
        return self.corruption.binomial(value, size=self.num_hid, n=1, p=o)

    def _corrupt_normal(self, value):
        """
            Corrupt with normal distribution
        """
        return self.corruption.gaussian(value, size=(self.num_hid,), avg=0.0, std=0.05)

    def _get_grads(self, cost):
        return T.grad(cost, self.params)

    def _penalty(self, x, xhat):
        """ Append additional penalty term to cost function(Check Sparse AE)
        """
        pass

    def _get_sample_from_trainingset(self, input_sample=True):
        if input_sample == True:
            sample_x = self.x[0]
        else:
            sample_x = self.x
        return sample_x

    def _get_sample(self, prob, lmb=0.01, input_sample=True, hidden_sample=False, idx=None, sample_data=None):
        """Sample from training examples x
           One step
           idx - index from training set
        """
        #sample_x_idx = theano.shared(self.numpy_rng.binomial(self.num_hid, 1))
        #MRG = RNG_MRG.MRG_RandomStreams(1)
        #sample = MRG.binomial(p = 0.5, size=(self.num_vis, self.num_hid,), dtype='float32')
        if sample_data == None:
            sample_x = self._get_sample_from_trainingset()
        else:
            # In this case, get all from data
            sample_x = sample_data
        hidden = T.dot(sample_x, self.W) + self.bh
        """ Corrupt this sample """
        corrupted_sample_x = T.nnet.sigmoid(self._corrupt_normal(hidden))
        if hidden_sample == False:
            # No sampels from hidden
            return sample_x, T.nnet.sigmoid(T.dot(corrupted_sample_x, self.W.T) + self.bv)

    def _reconstruct_sample(self, sample, input_sample=False):
        """ After getting sample, reconstruct it and get sample from reconstruction """
        return self._corrupt_normal(sample)

    def _compute_cost(self, all_loss, current_sample=None, decay=0.005):
        params = [self.W, self.bh]
        grad = T.grad(all_loss, params)
        return all_loss, [(param, param - decay * gradparam) for (param, gradparam) in zip(params, grad)]

        #cost = Loss().CrossEntropy(sample_x, reconstruct)
        # return cost
        """ Sample corrupted input """
        '''sample_corr = self.numpy_rng.binomial(self.num_hid, propb)
        sample_corrupted = T.dot(corrupted_sample_x, sample_corr)
        #Добавить этот сэмпл в скрытый слой (Приблизительная реализация)
        self.x[self.numvis + 1] = sample_corrupted
        #Cost need to be with negative log-likelihood
        cost = -T.mean(T.log(self.params) - lmb * self._penalty(sample_x, sample_corr))
        grads = self._get_grads(cost)
        #return cost, [(para, param - lrate * gparam) for (para, gparam) in zip(self.params, grads)]
        return cost'''

    def walkback_step(self, prob=None, p=0.5, distr=None):
        """ One step for walkback process
            distr - sample from this distribution
            Implementation from paper: Generalized Denoising Auto-Encoder as generative model
        """
        result = []
        ''' p - default probability '''
        if prob != None:
            ''' Sample from training examples X '''
            samplx_x_idx = self.numpy_rng.binomial(self.num_hid, prob)
        ''' In the case if prob = None, pick just one example from training set '''

        samplx_x_idx = self.x[np.random.randint(low=0, high=self.num_vis)]
        # Corrupt
        r = self.rng.binomial(self.numvis, p)
        samplx_x_corruptd = T.dot(self.x, r)
        u = np.random.random()
        if distr == 'geometric':
            u = np.random.geometric(p, size=1)
        if u > p:
            result.append(samplx_x_corruptd)

    def _corrupt_forward(self, sample):
        hidden = T.dot(sample, self.W) + self.bh
        hidden_corrupt = self._corrupt_normal(hidden)
        return T.nnet.sigmoid(hidden)

    def _reconstruct_forward(self, sample):
        return T.nnet.sigmoid(T.dot(sample, self.W.T) + self.bv)

    def walkback_step2(self, sample, steps=10):
        """
            sample - sample from training set
            Q - corrupt process
            P - reconstruction(not after each corruption step)
            Append additional training examples


            Need another cost
        """
        Q = self._corrupt_forward
        P = self._reconstruct_forward
        p = T.scalar(0.5)
        # Additional list of training examples
        result = []
        '''if sample == None:
            sample = self.x[np.random.randint(low=0, high=self.num_vis)]'''
        zvalue = sample
        for i in range(self.walkback_steps):
            zvalue = Q(zvalue)
        for i in range(self.walkback_steps):
            zvalue = Q(zvalue)
            zvalue = P(zvalue)
            #result.append((sample, zvalue))
        return sample, zvalue

    def walkback_step3(self, sample, steps, p=0.5):
        additional_samples = []
        Q = self._corrupt_forward
        P = self._reconstruct_forward
        zsample = sample
        for i in range(steps):
            corrupted = Q(zsample)
            additional_samples.append(corrupted)
            reconstruct = P(corrupted)
        return sample, additional_samples



        # Append (sample_x, sample_x_corrupted) as additional training example
        # In the last step, sample from P(X|\hat X*)

    '''def _walkbask_loss(self, elem1, elem2):
        return Loss().CrossEntropy(elem1, elem2)/elem1.shape[0]'''

    def _loss(self, elem1, elem2):
        return T.mean(T.nnet.binary_crossentropy(elem1, elem2))

    def train(self, propb, num_samples=10, momentum=0.9):
        """
            Main training phase
            num_samples - number of generated samples
        """
        # List of additional examples
        Addition = []
        # sample - "clean" sample
        # corruped - corrupted sample
        current_sample = None
        for i in range(5):
            if current_sample == None:
                current_sample = self.x
            corrx = self._corrupt_normal(current_sample)
            if self.walkback_steps > 0:
                sample, Addition = self.walkback_step3(current_sample, self.walkback_steps)
                corrupted = Addition[0]
            else:
                sample, corrupted = self._get_sample(
                    propb, sample_data=current_sample)
                Addition.append(corrupted)
            # Loss function over all generated samples
            # Need to append regularization
            reconstruction = self._reconstruct_sample(sample)

            all_loss = T.sum([self._loss(sample, gen) for gen in Addition])
            cost, updates = self._compute_cost(all_loss)

            '''func = theano.function(
                [self.x], [sample, corrupted, T.sum(self.W)])'''
            costfunc = theano.function(
                [], cost, updates=updates, givens={self.x: self.inp})
            print(costfunc())
            current_sample = corrupted

    def train_simple(self, prob, momentum=0.9, iters=5, num_samples=10, steps=10):
        """ Simple training of GSN without walkback and append noise during all of iterations"""
        for i in range(iters):
            Addition = []
            current_sample = None
            for step in range(steps):
                if current_sample == None:
                    current_sample = self.x
                corrupted = self._corrupt_forward(current_sample)
                reconstruction = self._reconstruct_forward(corrupted)
                Addition.append(reconstruction)
                current_sample = reconstruction
            all_loss = T.sum([self._loss(self.x, gen) for gen in Addition])
            cost, updates = self._compute_cost(all_loss)
            costfunc = theano.function(
                [], cost, updates=updates, givens={self.x: self.inp})
            print(costfunc())

    # After training, sample N examples
    # Generate from test set
    def sample(self, testset, N=10, sample=None):
        result = []
        for i in range(N - 1):
            corrupted = self.walkback_step2(sample, Q, P)
            result.append(corrupted)
        return result

    def _write(self, data, path):
        pass


class VariationalAutoencoder(AutoencoderPuppet):

    '''
        Auto-Encoding Variational Bayes
        http://arxiv.org/pdf/1312.6114v10.pdf
        Model for sampling
        generative model (encoder) and variational approximation (decoder)
    '''

    def __init__(self, x=None, theano_input=None, num_vis=100, num_hid=50, num_encoder=10, num_decoder=10, numpy_rng=None, lrate=0.001, corruption_level=0,
                 encoder_func='sigmoid', decoder_func=None, typeae='gaussian', encoder_type='gaussian', decoder_type='gaussian'):
        '''
            typeae is type for encoder/decoder
            gaussian - gaussian MLP for encoder-decoder
            bernoulli - bernoulli MLP for encoder
        '''
        AutoencoderPuppet.__init__(self, x=x, theano_input=theano_input, num_vis=num_vis, numpy_rng=numpy_rng, corruption_level=corruption_level,
                                   encoder_func=encoder_func, decoder_func=decoder_func)
        self.typeae = typeae
        self.num_hid = num_hid
        self.size = x.shape[0]
        self.encoder_type = encoder_type
        self.decoder_type = decoder_type
        par = ParametersInit(numpy_rng, -0.001, 0.001)
        self.W1 = par.get_weights((num_vis, num_encoder), 'W1')
        self.W2 = par.get_weights((num_encoder, num_hid), 'W2')
        self.W3 = par.get_weights((num_encoder, num_hid), 'W3')
        self.W4 = par.get_weights((num_hid, num_decoder), 'W4')
        self.W5 = par.get_weights((num_decoder, num_vis), 'W5')
        self.W6 = par.get_weights((num_decoder, num_vis), 'W6')
        self.b1 = theano.shared(
            np.asarray(np.zeros(num_encoder), dtype=theano.config.floatX), name='b1')
        self.b2 = theano.shared(
            np.asarray(np.zeros(num_hid), dtype=theano.config.floatX), name='b1')
        self.b3 = theano.shared(
            np.asarray(np.zeros(num_hid), dtype=theano.config.floatX), name='b3')
        self.b4 = theano.shared(
            np.asarray(np.zeros(num_decoder), dtype=theano.config.floatX), name='b4')
        self.b5 = theano.shared(
            np.asarray(np.zeros(num_vis), dtype=theano.config.floatX), name='b5')
        self.params = [
            self.W1, self.W2, self.W3, self.W4, self.W5, self.b1, self.b2]
        self.noise = T.matrix('noise')
        self.random_state = T.raw_random.random_state_type()

    def _costructAE(self, typeae):
        """ Construct for multi-layer version """
        pass

    def bern_decoder(self, reconstructed):
        """ Actually this is bernoulli decoder
            Bernoulli in case with binary data
        """
        return T.nnet.sigmoid(T.dot(T.tanh(T.dot(reconstructed, self.W2) + self.b2), self.W3) + self.b3)

    def gaussian_decoder(self, z):
        hidden = T.tanh(T.dot(z, self.W4) + self.b4)
        mu = T.dot(hidden, self.W5) + self.b5
        sigma = T.nnet.sigmoid(T.dot(hidden, self.W6))
        return hidden, mu, sigma

    def gaussian_encoder(self, x):
        mu = T.dot(x, self.W2)
        sigma = 0.5 * T.dot(x, self.W3)
        return mu, sigma

    def _cost(self):
        # Sample from noise distribution
        # For update need to use Adagrad instand SGD
        #self.typeae = 'bernoulli'

        # Compute MLP with or without dropout
        #Need to use x as a sample
        pre = T.nnet.sigmoid(T.dot(self.x, self.W1) + self.b1)
        if self.corruption_level > 0:
            binom = theano.shared(
                np.random.binomial(1, self.corruption_level, 10))
            encoder = pre * binom
        else:
            encoder = pre

        if self.encoder_type == 'gaussian':
            # Encoder
            mu, sigma = self.gaussian_encoder(encoder)
            noise = T.raw_random.normal(self.random_state, avg=0,std=0.001)
            prior = 1/2 * T.sum(1 + T.log(sigma**2) - mu**2 - sigma**2)
            z = mu + T.exp(sigma) * self.noise
            # With this values need to construct prior

            # z is sampling from normal distribution or exponential
        if self.decoder_type == 'gaussian':
            # Decoder
            result,mu_dec,sigmadec = self.gaussian_decoder(z)
            logrec = T.sum(T.log(sigmadec) - 1/2 * ((mu_dec/sigmadec))**2)
            cost = prior + logrec
            L = cost
        if self.decoder_type == 'bernoulli':
            hidden = self.bern_decoder(z)
            L = 0.005 * T.mean(Loss().CrossEntropy(self.x, hidden))
            #L = T.sum(hidden)
        epsilon = self.theano_rng.uniform(
            size=(self.num_hid, ), low=0.0, high=1.0)
        return L

    def train(self, batch_size=10):
        state = np.random.RandomState(1234)
        noise = np.random.random((self.size, self.num_hid))
        func = theano.function([self.x, self.noise], self._cost())
        print(func(self.training, noise))

    def _get_grads(self, cost):
        return T.grad(cost, self.params)


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
        value[d - 1] = 1
        corrm = T.shared(np.dot(value, value.T))
        prod = T.dot(self.x, self.x)
        Q = T.dot(prod, corrm)
        R = np.dot(value, T.diag(prod))
        rep = T.dot(prod, np.repeat(value, d))
        Wp = T.dot(self.W, self.W)
        hidden = T.tanh(np.dot(self.x, W))
        cost = T.mean(T.nnet.binary_crossentropy(hidden, self.x))
        grads = self._get_grads(cost)
        return cost, [(old,newparams - old * self.learning_rate) for (old, newparams) in zip(self.params, grads)]

    def _get_grads(self, cost):
        return T.sum(T.grad(self._cost, self.params))

    def train(self, p, iters=5000):
        for i in range(iters):
            cost = self._cost(p)
            grads = self._get_grads(cost)


def test_MarginalizedDAE():
    x = np.array([5, 4, 8, 5, 6, 9, 6, 3, 2, 1, 4, 8, 5, 8, 9])
    mda = MarginalizedDenoisingAutoencoder(x)
    mda.train(0.35)
    # Predict new values
    mda.predict()


class MADE(AutoencoderPuppet):

    ''' Masked Autoencoder for Distribution Estimation '''

    def __init__(self, num_vis, num_hid, x=None, theano_input=None, learning_rate=0.01, numpy_rng=None):
        if numpy_rng == None:
            numpy_rng = np.random.RandomState()
        par = ParametersInit(numpy_rng, 0.0, 1.1)
        self.W = par.get_weights(((num_vis, num_hid)))
        self.V = par.get_weights(((num_vis, num_hid)))
        self.biash = theano.shared(
            np.asarray(np.zeros(num_hid), dtype=theano.config.floatX), name='bh')
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
            maskhidden = theano.shared(
                np.random.randint(1, self.num_hid - 1, self.num_hid), name='maskh')
            maskvisible = theano.shared(
                np.random.randint(0, self.num_vis - 1, self.num_vis), name='maskv')
            return maskvisible, maskhidden
        elif num_layers > 1:
            maskhidden = []
            maskvisible = []
            for layer in range(num_layers):
                maskvisible.append(
                    T.shared(np.random.randint(1, self.num_hid - 1, self.num_hid), name='maskh'))
                maskhidden.append(
                    T.shared(np.random.randint(0, self.num_vis - 1, self.num_vis), name='maskv'))
            return maskvisible, maskhidden

    def _getBinaryMask(self, layers, num_layers):
        ''' 1 if mask[layer1[i]] > mask[layer0[i]] otherwise 0
            input - layers of mask, previusly generated from _getMask
        '''
        binary = T.matrix('bin')
        for layer in range(1, num_layers + 1):
            for i in range(layers[i].shape[0]):
                for j in range(layers[i].shape[1]):
                    pass

    def _cost(self):
        # Sample m^l vectors
        # get masks
        # MAskV, MaskH - должны быть бинарные матрицы
        MaskV, MaskH = self._getMask(1)
        hidden = T.dot(self.x, self.W * MaskH) + self.biash
        corrupt = T.nnet.sigmoid(T.dot(hidden, self.V * MaskV) + self.biash)
        cost = T.exp(
            T.sum(T.log(self.x * T.log(corrupt) + (1 - self.x) * T.log(1 - corrupt))))

        # Можно добавить Cross-entropy или Negative log likelihood
        grad = T.grad(-T.log(cost), self.params)
        # , [(param, param - self.learning_rate * gparam) for (param, gparam) in zip(self.params, grad)]
        return hidden[0][0]

    def _compute_gradients(self, corrupt, hidden):
        """ Compute gradients step-by-step """
        tmp = corrupt - self.x
        c_new = tmp
        V_new = T.dot(tmp, hidden) * self.V
        tmp = T.dot(tmp.T, self.V * self.W)
        # Compute with all layers

    def train(self):
        """ Main function of training """
        cost = self._cost()
        func = theano.function([self.x], cost)
        print(func(self.inp))
        '''func = theano.function([], cost, updates=updates, givens={self.x: self.inp})
        for i in range(10):
            print("cost: ", func())'''


class ContrastiveAutoEncoder(Autoencoder):

    """ Basic contrastive AE

        Higher Order Contractive Auto-Encoder
        http://www.iro.umontreal.ca/~vincentp/Publications/CAE_H_ECML2011.pdf
    """

    def __init__(self, x, theano_input=None, num_vis=100, num_hid=50, numpy_rng=None, lrate=0.001, corruption_level=0,
                 encoder_func='sigmoid', decoder_func=None):
        AutoencoderPuppet.__init__(self, x=x, theano_input=theano_input, num_vis=num_vis, numpy_rng=numpy_rng, corruption_level=corruption_level,
                                   encoder_func=encoder_func, decoder_func=decoder_func)
        self.lmb = lrate

    def jacobian(self, hidden, W):
        ''' trick from deeplearning.net tutorial '''
        return T.reshape(hidden * (1 - hidden), (self.num_vis, 1, self.num_hid)) * T.reshape(W, (1, self.num_vis, self.num_hid))

    def jacobian2(self, hidden, W):
        return T.sum((hidden * (1 - hidden)**2) * T.sum(W**2))

    def _cost(self, con_level=0.05):
        hidden = self.encoder(self.x, self.W)
        decode = self.decoder(hidden, self.W)
        #J = T.sum(self.jacobian(hidden, self.W)**2)/self.num_vis
        J = self.jacobian2(hidden, self.W) / self.num_vis
        L = T.mean(Loss().CrossEntropy(self.x, decode)) + con_level * T.mean(J)
        #L = T.sum(decode)
        grads = self._get_grads(L)
        return L, [(param, param - self.lrate * gradparam) for (param, gradparam) in zip(self.params, grads)]

    def train(self, iters=100):
        cost, updates = self._cost()
        func = theano.function(
            [], cost, updates=updates, givens={self.x: self.training})
        for i in range(iters):
            print(func())


class BlockedAutoencoder:

    ''' Кодируем и декодируем поблочно
        Разделяем входные юниты на n блоков, кодируем их(с разными степенями повреждения, замем, декодируем обратно)
     '''

    def __init__(self, x=None, theano_input=None, num_vis=100, num_hid=50, numpy_rng=None, lrate=0.001, corruption_level=0,
                 encoder_func='sigmoid', decoder_func=None):
        AutoencoderPuppet.__init__(self, x=x, theano_input=theano_input, num_vis=num_vis, numpy_rng=numpy_rng, corruption_level=corruption_level,
                                   encoder_func=encoder_func, decoder_func=decoder_func, block_num=5)
        self.lmb = lrate
        self.block_num = block_num

    def _encoder(self, value, weights):
        block = self.num_vis / block_num
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

    def train(self, batchtype='minibatch', batchsize=100, learning_type='sgd'):
        """ Start trainin model
            batchtype - (minibatch, online batch)
        """
        if learning_type == 'gll':
            ''' In this case, construct Stacked autoencoder '''
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
