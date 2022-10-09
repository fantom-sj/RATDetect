import tensorflow as tf
from tensorflow import keras
from keras import Model
import keras.backend as K
from keras.layers import Layer, Dense, GRU

import pandas as pd

from tensorflow.python.client import device_lib


class Encoder(Layer):
    def __init__(self, caracts_count, seed_kernel_init):
        super(Encoder, self).__init__()
        self.hidden_layer_1 = Dense(
            units=caracts_count,
            activation="relu",
            kernel_initializer=keras.initializers.he_uniform(seed_kernel_init)
        )
        self.hidden_layer_2 = Dense(
            units=caracts_count,
            activation="relu",
            kernel_initializer=keras.initializers.he_uniform(seed_kernel_init)
        )
        self.output_layer = Dense(
            units=caracts_count,
            activation="sigmoid"
        )

    def call(self, input_features):
        activation_1 = self.hidden_layer_1(input_features)
        activation_2 = self.hidden_layer_2(activation_1)
        return self.output_layer(activation_2)


class Decoder(tf.keras.layers.Layer):
    def __init__(self, caracts_count, seed_kernel_init):
        super(Decoder, self).__init__()
        self.hidden_layer_1 = Dense(
            units=caracts_count,
            activation="relu",
            kernel_initializer=keras.initializers.he_uniform(seed_kernel_init)
        )
        self.hidden_layer_2 = Dense(
            units=caracts_count,
            activation="relu",
            kernel_initializer=keras.initializers.he_uniform(seed_kernel_init)
        )
        self.output_layer = Dense(
            units=caracts_count,
            activation="sigmoid"
        )

    def call(self, code):
        activation_1 = self.hidden_layer_1(code)
        activation_2 = self.hidden_layer_2(activation_1)
        return self.output_layer(activation_2)


class Autoencoder(Model):
    def __init__(self, caracts_count, seed_kernel_init, batch_size):
        super(Autoencoder, self).__init__()
        self.batch_size = batch_size
        self.caracts_count = caracts_count
        self.encoder = Encoder(caracts_count=caracts_count, seed_kernel_init=seed_kernel_init)
        self.decoder = Decoder(caracts_count=caracts_count, seed_kernel_init=seed_kernel_init)

    def loss(self, x, y):
        x = K.reshape(x, shape=(self.batch_size, self.caracts_count))
        y = K.reshape(y, shape=(self.batch_size, self.caracts_count))
        loss = K.sum(K.square(x - y), axis=-1)
        return loss

    def call(self, input_features):
        code = self.encoder(input_features)
        reconstructed = self.decoder(code)
        return reconstructed


def normalization(pd_data):
    data_max = pd_data.max()
    data_min = pd_data.min()
    for col in pd_data:
        pd_data[col] = (pd_data[col] - data_min[col])/(data_max[col] - data_min[col])
    return pd_data


def main():
    data = pd.read_csv("C:\\Users\\Admin\\SnHome\\characts.csv")
    data = data.drop(["Time_Stamp"], axis=1)
    print("Загрузка датасета завершена.")

    data = normalization(data)
    print("Нормализация данных выполнена.")

    batch_size = 1
    epochs = 1
    caracts_count = 17
    seed = 89459876

    training_dataset = tf.convert_to_tensor(data.values)
    print("Обучающий датасет создан.")

    autoencoder = Autoencoder(caracts_count=caracts_count, seed_kernel_init=seed, batch_size=batch_size)
    autoencoder.compile(optimizer="adam", loss=autoencoder.loss)

    print("Начинаем обучание")
    autoencoder.fit(training_dataset, training_dataset, epochs=epochs, batch_size=batch_size, shuffle=False)
    autoencoder.save("model_Dense")






if __name__ == '__main__':
    # print(device_lib.list_local_devices())
    main()
