from typing import Tuple

import tensorflow as tf
from keras.api._v2 import keras
from keras.utils import to_categorical
from nptyping import NDArray, Shape, Int, Float


def preprocess_dataset(
        images: NDArray[Shape['Sample, Width, Height'], Int], 
        labels: NDArray[Shape['Sample'], Int]
    ) -> Tuple[NDArray[Shape['Sample, Width_x_Height'], Int], NDArray[Shape['Sample, Class'], Int]]:
    images: NDArray[Shape['Sample, Width_x_Height']] = images.reshape((images.shape[0], -1))
    labels: NDArray[Shape['Sample, Class']] = to_categorical(labels)
    return images, labels
