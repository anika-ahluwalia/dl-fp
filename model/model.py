import tensorflow as tf
import numpy as np
from tensorflow.keras import Model

class RestorationModel(tf.keras.Model):
    def __init__(self):
        super(RestorationModel, self).__init__()
    
    @tf.function
    def call(self):
        pass

    def loss(self):
        pass

def main():
    model = RestorationModel()

if __name__ == '__main__':
    main()