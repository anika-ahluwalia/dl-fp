import tensorflow as tf
import numpy as np
from tensorflow.keras import Model

class Analysis(tf.keras.Model):
    def __init__(self):
        super(Analysis, self).__init__()
    
    @tf.function
    def call(self):
        pass

    def loss(self):
        pass

def main():
    model = Analysis()

if __name__ == '__main__':
    main()