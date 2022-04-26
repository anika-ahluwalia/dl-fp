import tensorflow as tf
import numpy as np
from tensorflow.keras import Model

class RestorationModel(tf.keras.Model):
    def __init__(self):
        super(Model, self).__init__()
    
    def call(self):
        pass

    def loss(self):
        pass

def train(model, train_inputs, train_labels):
    pass

def test(model, test_inputs, test_labels):
    pass

def main():
    model = RestorationModel()
    train(model, None, None)
    test(model, None, None)

if __name__ == '__main__':
    main()