"""
    Модуль в котором содержится архитектура рекуррентной нейронной сети,
    используемой для детектирования аномалий в сетевом трафике.
"""

import tensorflow as tf
from tensorflow import keras
from keras.layers import GRU, LSTM
from AnomalyDetector.AutoEncoder import Autoencoder_Base

import pandas as pd
import numpy as np


class Autoencoder(Autoencoder_Base):
    """
        Класс описывающий из чего состоит автоэнкодер и как
        происходит его обучение на каждом шаге, какие вычисляются метрики
        и как происходит обратное распространение ошибки для обучения.
    """

    def __init__(self, caracts_count: int, arhiteche: dict, window_size=1000):
        super(Autoencoder, self).__init__()
        self.window_size = window_size
        self.caracts_count = caracts_count

        self.encdec = []
        for layer in arhiteche:
            units, input_size = arhiteche[layer]
            if "GRU" in layer:
                self.encdec.append(
                    GRU(
                        units=units,
                        activation="tanh",
                        return_sequences=True,
                        name=f"layer_t.{layer}_type.{units}_to_{input_size}",
                        input_shape=(self.window_size, input_size),
                        # dropout=0.2
                        # stateful=True,
                        return_state=True
                    )
                )
            elif "LSTM" in layer:
                self.encdec.append(
                    LSTM(
                        units=units,
                        activation="tanh",
                        return_sequences=True,
                        name=f"layer_t.{layer}_type.{units}_to_{input_size}",
                        input_shape=(self.window_size, input_size),
                        # dropout=0.2
                        # stateful=True,
                        return_state=True
                    )
                )

        self.loss_tracker = keras.metrics.Mean(name="loss")
        self.mae_metric = keras.metrics.MeanAbsoluteError(name="mae")

        self.valid_loss_tracker = keras.metrics.Mean(name="valid_loss")
        self.valid_mae_metric = keras.metrics.MeanAbsoluteError(name="valid_mae")

        self.history_loss = {"epoch": [], "step": [], "loss": [], "mean_loss": [], "mae": []}
        self.history_valid = {"epoch": [], "step": [], "loss": [], "mean_loss": [], "mae": []}

    def call(self, input_features):
        x = input_features
        for layer in self.encdec:
            x, state = layer(x)
        return x


class TrainingDatasetGen(keras.utils.Sequence):
    """
        Класс генератора нормализованных данных для обучения нейронной сети
        автоэнкодера. Позволяет генерировать данные наборами batch_size,
        чтобы в дальнейшем передавать по одному батчу в нейронную сеть для очередного шага обучения.
    """

    def __init__(self, dataset, max_min_file, feature_range, batch_size=1000, windows_size=1000, validation_factor=0.2):
        # Нормализуем данные
        self.dataset = self.normalization(dataset, max_min_file, feature_range).to_numpy() #[:5000]
        if np.isnan(np.sum(self.dataset)):
            print("В обучающем датасете были обнаружены nan-данные, они были заменены на 0")
            self.dataset = np.nan_to_num(self.dataset)

        print("Нормализация данных выполнена.")

        self.numbs_count, self.caracts_count = self.dataset.shape

        self.windows_size = windows_size
        self.batch_size = batch_size

        # Получаем размеры тренировочной и валидационной выборки
        self.valid_count = round(self.numbs_count * validation_factor / batch_size) * batch_size
        self.numbs_count = round(len(self.dataset) / batch_size) * batch_size - self.valid_count

        # Создаём тренировочную и валидационную выборку
        self.training_dataset = self.dataset[:self.numbs_count]
        self.training_dataset = tf.convert_to_tensor(self.training_dataset)
        self.training_dataset = tf.reshape(self.training_dataset,
                                           (round(self.numbs_count / self.batch_size), self.windows_size,
                                            self.caracts_count))
        self.valid_dataset = self.dataset[self.numbs_count:round(len(self.dataset) / batch_size) * batch_size]

    @staticmethod
    def normalization(pd_data, max_min_file=None, feature_range=(0, 1), mix_max_from_file=False):
        if mix_max_from_file:
            read_min_max = pd.read_csv(max_min_file)
            data_max     = read_min_max.iloc[0]
            data_min     = read_min_max.iloc[1]
            print(read_min_max)
        else:
            data_max = pd_data.max()
            data_min = pd_data.min()

        if (not mix_max_from_file) and (max_min_file is not None):
            cols_name = []
            for col in pd_data:
                cols_name.append(col)
            pd_ch_name = pd.DataFrame([data_max, data_min], columns=cols_name)
            print(pd_ch_name)
            pd_ch_name.to_csv(max_min_file, index=False)

        min_f, max_f = feature_range
        for col in pd_data:
            if col != "Time_Stamp":
                pd_data[col] = (pd_data[col] - data_min[col]) / (data_max[col] - data_min[col])
                pd_data[col] = pd_data[col] * (max_f - min_f) + min_f
        return pd_data

    def __len__(self):
        return round(self.numbs_count / self.batch_size)

    def __getitem__(self, idx):
        batch_x = tf.gather(self.training_dataset, idx, axis=0)
        batch_x = tf.reshape(batch_x, (1, self.windows_size, self.caracts_count))
        return batch_x

    def get_valid_len(self):
        return round((self.valid_count - self.windows_size) / self.batch_size)

    def get_valid(self):
        valid_arr = []
        len_valid = self.get_valid_len()
        for idx in range(len_valid):
            valid_batch_x = np.array(
                [self.valid_dataset[idx * self.batch_size:idx * self.batch_size + self.windows_size, :]])
            valid_arr.append(tf.convert_to_tensor(valid_batch_x))
        return valid_arr

    def on_epoch_end(self):
        print("Перемешивание обучающего датасета!")
        self.training_dataset = tf.random.shuffle(self.training_dataset)
        print("Перемешивание выполнено.")
