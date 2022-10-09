"""
    Модуль в котором содержится архитектура рекуррентной нейронной сети,
    используемой для детектирования аномалий в сетевом трафике.
"""

import tensorflow as tf
from tensorflow import keras
from keras import Model, Sequential
from keras.layers import Layer, GRU, InputLayer

import pandas as pd
import numpy as np
import random

from tensorflow.python.client import device_lib


class DecEncoder(Layer):
    """
        Класс описывающий внутреннюю структуру автоэнкодера,
        а именно какие в нем имеются скрытые слои и как они
        взаимодействуют друг с другом.
    """

    def __init__(self, input_shape, type, seed_kernel_init, count_hidden_layers):
        super(DecEncoder, self).__init__()
        self.input_layer = InputLayer(input_shape)
        self.batch_size, self.windows_size, self.caracts_count = input_shape
        self.count_hidden_layers = count_hidden_layers
        self.seed_kernel_init = seed_kernel_init

        self.hidden_layers = Sequential()
        self.output_layer = Sequential()

        if type == "encoder":
            start = 0
            end = self.count_hidden_layers * (-1)
            step = -1
        else:
            start = (self.count_hidden_layers-1) * (-1)
            end = 1
            step = 1

        index = 0
        for i in range(start, end, step):
            self.hidden_layers.add(
                GRU(
                    units=self.caracts_count + i,
                    activation="tanh",
                    kernel_initializer=keras.initializers.he_uniform(self.seed_kernel_init),
                    return_sequences=True,
                    name="hidden_layer_" + str(index),
                    input_shape=(self.windows_size, self.caracts_count + i)
                )
            )
            index += 1

    def call(self, input_characts):
        activation = self.input_layer(input_characts)
        activation = self.hidden_layers(activation)
        activation = self.output_layer(activation)
        return activation


class Autoencoder(Model):
    """
        Класс описывающий из чего состоит автоэнкодер и как
        происходит его обучение на каждом шаге, какие вычисляются метрики
        и как происходит обратное распространение ошибки для обучения.
    """

    def __init__(self, caracts_count=None, seed_kernel_init=231654789,
                 batch_size=1, windows_size=1000, count_hidden_layers=1):
        super(Autoencoder, self).__init__()
        self.batch_size = batch_size
        self.caracts_count = caracts_count
        self.encoder = DecEncoder(input_shape=(batch_size, windows_size, caracts_count), type="encoder",
                                  seed_kernel_init=seed_kernel_init, count_hidden_layers=count_hidden_layers)
        self.decoder = DecEncoder(input_shape=(batch_size, windows_size, caracts_count), type="decoder",
                                  seed_kernel_init=seed_kernel_init, count_hidden_layers=count_hidden_layers)

        self.loss_tracker = keras.metrics.Mean(name="loss")
        self.mae_metric = keras.metrics.MeanAbsoluteError(name="mae")

    def train_step(self, data):
        x = data

        with tf.GradientTape() as tape:
            y_pred = self(x, training=True)
            loss = keras.losses.kl_divergence(x, y_pred)
            # loss = keras.losses.kl_divergence(x, y_pred)

        # Вычисляем градиент
        trainable_vars = self.trainable_variables
        gradients = tape.gradient(loss, trainable_vars)

        # Обновляем веса
        self.optimizer.apply_gradients(zip(gradients, trainable_vars))

        # Расчитываем выходные метрики
        self.loss_tracker.update_state(loss)
        self.mae_metric.update_state(x, y_pred)

        return {"Расхождение": loss, "Средние расхождение": self.loss_tracker.result(),
                "Средняя абсолютная ошибка": self.mae_metric.result()}

    def call(self, input_features):
        code = self.encoder(input_features)
        reconstructed = self.decoder(code)
        return reconstructed

    @property
    def metrics(self):
        return [self.loss_tracker, self.mae_metric]


class TrainingDatasetGen(keras.utils.Sequence):
    """
        Класс генератора нормализованных данных для обучения нейронной сети
        автоэнкодера. Позволяет генерировать данные наборами batch_size,
        чтобы в дальнейшем передавать по одному батчу в нейронную сеть для очередного шага обучения.
    """

    def __init__(self, training_dataset, batch_size, windows_size, max_min_file):
        self.training_dataset = self.normalization(training_dataset, max_min_file).to_numpy()
        self.training_dataset = self.training_dataset[:round(len(self.training_dataset)/batch_size)*batch_size]
        print("Нормализация данных выполнена.")

        self.numbs_count, self.caracts_count = self.training_dataset.shape
        self.windows_size = windows_size
        self.batch_size = batch_size

    def normalization(self, pd_data, max_min_file):
        data_max = pd_data.max()
        data_min = pd_data.min()
        cols_name = []
        for col in pd_data:
            cols_name.append(col)
        pd_ch_name = pd.DataFrame([data_max, data_min], columns=cols_name)
        print(pd_ch_name)
        pd_ch_name.to_csv(max_min_file, index=False)

        for col in pd_data:
            if col != "Time_Stamp":
                pd_data[col] = (pd_data[col] - data_min[col]) / (data_max[col] - data_min[col])
        return pd_data

    def __len__(self):
        return round((self.numbs_count - self.windows_size)/self.batch_size)

    def __getitem__(self, idx):
        batch_x = np.array([self.training_dataset[idx*self.batch_size:idx*self.batch_size + self.windows_size, :]])
        return batch_x


def main():
    data = pd.read_csv("F:\\VNAT\\itog_video.csv")
    max_min_file = "max_and_min_video.csv"

    data = data.drop(["Time_Stamp"], axis=1)
    print("Загрузка датасета завершена.")

    batch_size = 1000
    count_hidden_layers = 7
    seed = random.randint(3654756461, 9834548734)
    windows_size = 1000
    epochs = 5
    model_name = "model_GRU_VNAT_video"

    training_dataset = TrainingDatasetGen(data, batch_size, windows_size, max_min_file)
    print(training_dataset.numbs_count, training_dataset.caracts_count)
    print("Обучающий датасет создан.")

    autoencoder = Autoencoder(caracts_count=training_dataset.caracts_count, seed_kernel_init=seed,
                              batch_size=batch_size, windows_size=windows_size,
                              count_hidden_layers=count_hidden_layers)
    autoencoder.compile(optimizer="adam")

    print("Автоэнкодер определён.")

    print("Начинаем обучение:")
    history = autoencoder.fit(training_dataset, epochs=epochs, batch_size=batch_size, shuffle=True, verbose=1)
    # autoencoder_load = keras.models.load_model(model_name)
    # autoencoder_load.compile(optimizer="adam", loss="kdl")
    # history = autoencoder_load.fit(training_dataset, epochs=epochs, batch_size=batch_size, shuffle=True, verbose=1)
    autoencoder.save(model_name)
    autoencoder.summary()
    autoencoder.encoder.hidden_layers.summary()
    autoencoder.decoder.hidden_layers.summary()

    # print(history.history)
    pd.DataFrame(history.history).to_csv("History_train_VNAT_3.csv", index=False)


if __name__ == '__main__':
    # print(device_lib.list_local_devices())
    main()
