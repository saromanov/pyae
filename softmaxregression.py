import theano.tensor as T
import theano
import numpy as np
from parametersinit import ParametersInit

class SoftmaxRegression:

    """ Basic implementation of softmax regression(for Stacked Autoencoder)
        http://ufldl.stanford.edu/wiki/index.php/Softmax_Regression

        http://blog.datumbox.com/machine-learning-tutorial-the-multinomial-logistic-regression-softmax-regression/
    """

    def __init__(self, inp=None, y=None, hid_num=30, num_classes=3, numpy_rng=None, theano_inp=None, theano_labels=None, lrate=0.1, inp_num=64):
        self.inp = inp
        self.labels = y
        self.num_classes = num_classes
        self.size = num_classes
        if theano_inp == None:
            self.x = T.matrix('x')
        else:
            self.x = theano_inp
        self.y = theano_labels
        if theano_labels == None:
            self.y = T.ivector('y')
        if numpy_rng == None:
            self.rng = np.random.RandomState(55)
        else:
            self.rng = numpy_rng
        par = ParametersInit(self.rng, -0.0001, 0.0001)
        self.W = par.get_weights((inp_num, hid_num), 'Wreg')
        self.bh = theano.shared(
            np.asarray(np.zeros(hid_num), dtype=theano.config.floatX), name='bhreg')
        self.lrate = lrate
        self.params = [self.W, self.bh]

    def _get_grads(self, cost):
        return T.grad(cost, self.params)

    def add_theano_input(self, x):
        self.x = x

    def updateParams(self, newparams):
        """ Update params during training with many layers """
        for i in range(len(newparams)):
            self.params[i] = newparams[i]

    def forward(self):
        value = T.nnet.softmax(T.dot(self.x, self.W) + self.bh)
        prediction = T.argmax(value, axis=1)
        return value, prediction
        #return T.sum(T.sum(T.log(value), axis=0)) * T.sum(T.log(self.y)), prediction

    def cost(self, weight_decay=0.9):
        value, prediction = self.forward()
        #result = -T.mean(T.log(value)[T.arange(self.y.shape[0]), self.y])
        result = T.sum(self.y * T.log(value) + (1 - self.y * T.log(1 - value)))
        return result
        #grads = self._get_grads(result)
        #return result, [(param, param - self.lrate * gparam) for param, gparam in zip(self.params, grads)]

    def global_error(self, pred):
        return T.mean(T.neq(pred, self.y))

    def train(self):
        cost_value, updates = self._cost()
        #func = theano.function([], cost, updates=updates, givens={self.x: self.inp, self.y: self.labels})
        func = theano.function(
            [], cost_value, updates=updates, givens={self.x: self.inp, self.y: self.labels})
        for i in range(100):
            print("iter {0}".format(i))
            print(func())


