import numpy as np
from sklearn.cross_validation import train_test_split

#Construct dataset

class Dataset:
    def __init__(self, X, y):
        self.X = X
        self.y = y

    def split(test_size=.25, random_state=123):
        return train_test_split(self.X, self.y, test_size=test_size, random_state=random_state
