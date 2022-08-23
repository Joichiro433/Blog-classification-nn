import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from keras.api._v2 import keras
from keras.datasets import mnist
from rich import print

import params
from preprocessing import preprocess_dataset


def predict(images):
    model = keras.models.load_model(params.MODEL_FILE_PATH)
    pred = model.predict(images)
    return np.argmax(pred, axis=1)


if __name__ == '__main__':
    (train_images, train_labels), (test_images, test_labels) = mnist.load_data()
    test_images, test_labels = test_images[:10], test_labels[:10]  # 10枚だけ推論をおこなう
    test_images = preprocess_dataset(images=test_images)
    pred = predict(test_images)
    print(f'prediction: {pred}')
    print(f'labels: {test_labels}')
    # 推論した画像を表示する
    for idx, image in enumerate(test_images):
        plt.subplot(1, 10, idx+1)
        plt.imshow(image.reshape((28, 28)), 'gray')
        plt.axis("off")    
    plt.show()
