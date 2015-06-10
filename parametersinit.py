import theano
import numpy as np

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

    def get_weights(self, size, name, init_type='standard'):
        """
            size in tuple format
            name - current name for weights
        """
        if init_type == 'xavier':
            return self._initW2(size, self.low, self.high, name)
        else:
            return self._initW(size, name)

    def _initW(self, size, name):
        return theano.shared(value=np.asarray(
            self.rng.uniform(low=self.low, high=self.high, size=size), dtype=theano.config.floatX
        ))

    def _initW2(self, size, nin, nout, name):
        return theano.shared(value=np.asarray(
            self.rng.uniform(low=-np.sqrt(6) / np.sqrt(nin + nout), high=np.sqrt(6) / np.sqrt(nin + nout),
                             size=size), dtype=theano.config.floatX
        ), name=self.name)


