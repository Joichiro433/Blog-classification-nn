import tensorflow as tf
from keras.api._v2 import keras
from keras.models import Model
from keras.layers import Input, Dense, Dropout

import params


class DNNModel:
    def __init__(
            self,
            input_size: int = params.INPUT_SIZE,
            hidden1_size: int = params.HIDDEN1_SIZE,
            hidden2_size: int = params.HIDDEN2_SIZE,
            output_size: int = params.OUTPUT_SIZE,
            dropout_rate: float = 0.5) -> None:
        self.input: Input = Input(shape=(input_size,), name='input')
        self.hidden1: Dense = Dense(hidden1_size, activation='relu', name='hidden1')
        self.hidden2: Dense = Dense(hidden2_size, activation='relu', name='hidden2')
        self.dropout: Dropout = Dropout(rate=dropout_rate, name='dropout')
        self.output: Dense = Dense(output_size, activation='softmax', name='output')

    def build(self) -> Model:
        input = self.input
        x = self.hidden1(input)
        x = self.hidden2(x)
        x = self.dropout(x)
        output = self.output(x)
        return Model(inputs=input, outputs=output)
        