import tensorflow as tf
import numpy as np
from tensorflow.keras import Model

class AnalysisModel(tf.keras.Model):
    def __init__(self):
        super(AnalysisModel, self).__init__()
    
    @tf.function
    def call(self):
        pass

    def loss(self):
        pass