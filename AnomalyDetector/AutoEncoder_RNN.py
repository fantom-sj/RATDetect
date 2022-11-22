"""
    Модуль в котором содержится архитектура рекуррентной нейронной сети,
    используемой для детектирования аномалий в сетевом трафике.
"""
from keras.layers import GRU, LSTM, Input, RepeatVector, TimeDistributed, Dense
from AutoEncoder import AutoencoderBase
from tensorflow import keras
from keras.models import Sequential
from tqdm import tqdm

import tensorflow as tf
import pandas as pd
import numpy as np
import math


class Autoencoder(AutoencoderBase):
    """
        Класс описывающий из чего состоит автоэнкодер и как
        происходит его обучение на каждом шаге, какие вычисляются метрики
        и как происходит обратное распространение ошибки для обучения.
    """

    def __init__(self, caracts_count: int, arhiteche: dict, window_size=1000):
        super(Autoencoder, self).__init__()
        self.window_size = window_size
        self.caracts_count = caracts_count

        self.graph = Sequential(name="Graph_network")
        for layer in arhiteche:
            shape_1, shape_2 = arhiteche[layer]

            if "Input" in layer:
                self.graph.add(
                    Input(
                        shape=(shape_1, shape_2),
                        name=f"layer_t.{layer}_shape.{shape_1}_{shape_2}"
                    )
                )
                print("add input")
            if "GRU" in layer:
                if "seq" in layer:
                    return_sequences = True
                else:
                    return_sequences = False
                self.graph.add(
                    GRU(
                        units=shape_1,
                        activation="tanh",
                        return_sequences=return_sequences,
                        name=f"layer_t.{layer}_type.{shape_1}_to_{shape_2}",
                        input_shape=(self.window_size, shape_2),
                        # stateful=True,
                    )
                )
            elif "LSTM" in layer:
                if "seq" in layer:
                    return_sequences = True
                else:
                    return_sequences = False
                self.graph.add(
                    LSTM(
                        units=shape_1,
                        activation="tanh",
                        return_sequences=return_sequences,
                        name=f"layer_t.{layer}_type.{shape_1}_to_{shape_2}",
                        input_shape=(self.window_size, shape_2),
                        # stateful=True,
                    )
                )
            elif "RepeatVector" in layer:
                self.graph.add(
                    RepeatVector(shape_1),
                )
            elif "TimeDistributed" in layer:
                self.graph.add(
                    TimeDistributed(Dense(shape_2))
                )

        self.loss_tracker = keras.metrics.Mean(name="loss")
        self.mae_metric = keras.metrics.MeanAbsoluteError(name="mae")

        self.valid_loss_tracker = keras.metrics.Mean(name="valid_loss")
        self.valid_mae_metric = keras.metrics.MeanAbsoluteError(name="valid_mae")

        self.history_loss = {"epoch": [], "step": [], "loss": [], "mean_loss": [], "mae": []}
        self.history_valid = {"epoch": [], "step": [], "loss": [], "mean_loss": [], "mae": []}

    def call(self, input_features):
        return self.graph(input_features)


class TrainingDatasetGen(keras.utils.Sequence):
    """
        Класс генератора нормализованных данных для обучения нейронной сети
        автоэнкодера. Позволяет генерировать данные наборами batch_size,
        чтобы в дальнейшем передавать по одному батчу в нейронную сеть для очередного шага обучения.
    """

    def __init__(self, dataset, max_min_file, feature_range, cp=0,
                 batch_size=1000, windows_size=1000, validation_factor=0.2):
        
        # Нормализуем данные
        sdvig = cp*10
        self.dataset = self.normalization(dataset, max_min_file, feature_range).to_numpy()[sdvig:]
        print(f"Данные сдвинуты на {sdvig}")
        if np.isnan(np.sum(self.dataset)):
            print("В обучающем датасете были обнаружены nan-данные, они были заменены на 0")
            self.dataset = np.nan_to_num(self.dataset)

        print("Нормализация данных выполнена.")

        self.numbs_count, self.caracts_count = self.dataset.shape

        self.windows_size       = windows_size
        self.batch_size         = batch_size
        self.validation_factor  = validation_factor

        # Получаем размеры тренировочной и валидационной выборки
        self.valid_count = math.floor(self.numbs_count * self.validation_factor / self.windows_size) * self.windows_size

        # Получаем количество характеристик за вычетом валидационной выборки и количество порций данных
        self.numbs_count = math.floor(math.floor(self.numbs_count / self.windows_size) / self.batch_size) * \
                           self.windows_size * self.batch_size - self.valid_count
        self.count_batch_training_dataset = math.floor(math.floor(self.numbs_count / self.windows_size) / self.batch_size)
        self.numbs_count = self.count_batch_training_dataset * self.batch_size * self.windows_size

        # Создаём тренировочную и валидационную выборку
        self.training_dataset = self.dataset[:self.numbs_count]
        self.training_dataset = tf.convert_to_tensor(self.training_dataset)
        self.training_dataset = tf.reshape(self.training_dataset,
                                           (self.count_batch_training_dataset,
                                            self.batch_size,
                                            self.windows_size,
                                            self.caracts_count))

        self.valid_dataset = self.dataset[self.numbs_count: self.numbs_count + self.valid_count]
        self.valid_dataset = tf.convert_to_tensor(self.valid_dataset)
        self.valid_dataset = tf.reshape(self.valid_dataset,
                                        (math.floor(self.valid_count / self.windows_size),
                                         1,
                                         self.windows_size,
                                         self.caracts_count))

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

        caracts_count = len(pd_data.axes[1])
        pbar = tqdm(total=caracts_count, desc="Прогресс нормализации")
        for col in pd_data:
            pd_data[col] = (pd_data[col] - data_min[col]) / (data_max[col] - data_min[col])
            pd_data[col] = pd_data[col] * (max_f - min_f) + min_f
            pbar.update(1)
        pbar.close()

        return pd_data

    def __len__(self):
        return math.floor(math.floor(self.numbs_count / self.windows_size) / self.batch_size)

    def __getitem__(self, idx):
        batch_x = tf.gather(self.training_dataset, idx, axis=0)
        return batch_x

    def get_valid_len(self):
        return math.floor((self.valid_count - self.windows_size) / self.windows_size)

    def get_valid(self):
        len_valid = self.get_valid_len()
        for idx in range(len_valid):
            valid_batch_x = tf.gather(self.valid_dataset, idx, axis=0)
            yield valid_batch_x

    def on_epoch_end(self):
        print("Сдвиг обучающего датасета!")
        self.dataset = self.dataset[10:]
        print(f"Данные сдвинуты на {10}")

        self.numbs_count, self.caracts_count = self.dataset.shape

        # Получаем размеры тренировочной и валидационной выборки
        self.valid_count = math.floor(self.numbs_count * self.validation_factor / self.windows_size) * self.windows_size

        # Получаем количество характеристик за вычетом валидационной выборки и количество порций данных
        self.numbs_count = math.floor(math.floor(self.numbs_count / self.windows_size) / self.batch_size) * \
                           self.windows_size * self.batch_size - self.valid_count
        self.count_batch_training_dataset = math.floor(
            math.floor(self.numbs_count / self.windows_size) / self.batch_size)
        self.numbs_count = self.count_batch_training_dataset * self.batch_size * self.windows_size

        # Создаём тренировочную и валидационную выборку
        self.training_dataset = self.dataset[:self.numbs_count]
        self.training_dataset = tf.convert_to_tensor(self.training_dataset)
        self.training_dataset = tf.reshape(self.training_dataset,
                                           (self.count_batch_training_dataset,
                                            self.batch_size,
                                            self.windows_size,
                                            self.caracts_count))
        print("Сдвиг выполнен выполнено.")
