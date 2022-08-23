from typing import Tuple, Optional, Union

import tensorflow as tf
from keras.api._v2 import keras
from keras.utils import to_categorical
from nptyping import NDArray, Shape, Int, Float


# 型情報
Images = NDArray[Shape['Sample, Width, Height'], Int]
Labels = NDArray[Shape['Sample'], Int]
PImages = NDArray[Shape['Sample, Width_x_Height'], Int]
PLabels = NDArray[Shape['Sample, Class'], Int]


def preprocess_dataset(
        images: Images, 
        labels: Optional[Labels] = None) -> Union[PImages, Tuple[PImages, PLabels]]:
    images: PImages = images.reshape((images.shape[0], -1))
    if labels is None:
        return images
    labels: PLabels = to_categorical(labels)
    return images, labels
