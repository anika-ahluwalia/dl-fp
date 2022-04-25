import os
from typing import Tuple, List
import tensorflow as tf


def preprocess(file_path: str):
    """
    Transforms files at file_path from PNG to Tensors, one for training and one for testing. Adds random noise to both
    sets of images, with original images as labels.

    :param file_path: Path to the folder containing the PNG images.
    :return: Two datasets, the first for training and the second for testing.
    """
    noise_layer = tf.keras.layers.GaussianNoise(0.01) # might need to change this value later

    image_tensors: List[tf.Tensor] = []
    for image in os.listdir(file_path):
        if image.endswith(".png"):
            image_tensor = tf.io.decode_image(image)
            image_tensors.append(image_tensor)
    training_data = image_tensors[:-17500]

    for i in range(len(training_data)):
        image_with_noise = noise_layer(training_data[i])
        training_data[i] = image_with_noise
    training_labels = image_tensors[:-17500]

    testing_data = image_tensors[52500:]

    for i in range(len(testing_data)):
        image_with_noise = noise_layer(testing_data[i])
        testing_data[i] = image_with_noise

    testing_labels = image_tensors[52500:]

    training_dataset = tf.data.Dataset.from_tensors((training_data, training_labels))
    testing_dataset = tf.data.Dataset.from_tensors((testing_data, testing_labels))

    return training_dataset, testing_dataset
