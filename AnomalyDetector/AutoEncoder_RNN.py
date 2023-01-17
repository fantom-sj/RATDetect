"""
    Модуль в котором содержится архитектура рекуррентной нейронной сети,
    используемой для детектирования аномалий в сетевом трафике.
"""
import array

from keras.layers import GRU, LSTM, Input, RepeatVector, TimeDistributed, Dense, Lambda, Dropout, BatchNormalization
from AnomalyDetector.AutoEncoder import AutoencoderBase, noiser
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

    def __init__(self, characts_count: int, arhiteche: (), window_size=1000, batch_size=1):
        super(Autoencoder, self).__init__()
        self.batch_size     = batch_size
        self.window_size    = window_size
        self.characts_count = characts_count

        encoder, decoder = arhiteche
        self.encoder_model = self.createSubModel(encoder, "encoder_model")
        self.decoder_model = self.createSubModel(decoder, "decoder_model")

        self.loss_tracker = keras.metrics.Mean(name="loss", dtype="float32")
        self.mae_metric = keras.metrics.MeanAbsoluteError(name="mae", dtype="float32")

        self.valid_loss_tracker = keras.metrics.Mean(name="valid_loss", dtype="float32")
        self.valid_mae_metric = keras.metrics.MeanAbsoluteError(name="valid_mae", dtype="float32")

    def createSubModel(self, arhiteche, name):
        sub_model = Sequential(name=name)
        for layer in arhiteche:
            shape_1, shape_2 = arhiteche[layer]

            if "Input" in layer:
                sub_model.add(
                    Input(
                        # shape=(shape_1, shape_2),
                        batch_input_shape=[self.batch_size, shape_1, shape_2],
                        name=f"layer_t.{layer}_shape.{shape_1}_{shape_2}",
                        dtype="float32"
                    )
                )
            if "GRU" in layer:
                if "seq" in layer:
                    return_sequences = True
                else:
                    return_sequences = False
                sub_model.add(
                    GRU(
                        units=shape_1,
                        activation="tanh",
                        return_sequences=return_sequences,
                        name=f"layer_t.{layer}_type.{shape_1}_to_{shape_2}",
                        input_shape=(self.window_size, shape_2),
                        dtype="float32",
                        stateful=True,
                    )
                )
                sub_model.add(BatchNormalization(dtype="float32"))
                sub_model.add(Dropout(0.25, dtype="float32"))
            elif "LSTM" in layer:
                if "seq" in layer:
                    return_sequences = True
                else:
                    return_sequences = False
                sub_model.add(
                    LSTM(
                        units=shape_1,
                        activation="tanh",
                        return_sequences=return_sequences,
                        name=f"layer_t.{layer}_type.{shape_1}_to_{shape_2}",
                        input_shape=(self.window_size, shape_2),
                        dtype="float32",
                        stateful=True,
                    )
                )
                sub_model.add(BatchNormalization(dtype="float32"))
                sub_model.add(Dropout(0.25, dtype="float32"))
            elif "RepeatVector" in layer:
                sub_model.add(
                    RepeatVector(shape_1, dtype="float32"),
                )
            elif "TimeDistributed" in layer:
                sub_model.add(
                    TimeDistributed(Dense(shape_2, dtype="float32"))
                )
        return sub_model

    def call(self, input_features, **kwargs):
        encoder_res = self.encoder_model(input_features)
        decoder_res = self.decoder_model(encoder_res)
        return decoder_res


class TrainingDatasetGen(keras.utils.Sequence):
    """
        Класс генератора нормализованных данных для обучения нейронной сети
        автоэнкодера. Позволяет генерировать данные наборами batch_size,
        чтобы в дальнейшем передавать по одному батчу в нейронную сеть для очередного шага обучения.
    """

    def __init__(self, dataset, max_min_file, feature_range,
                 batch_size=1000, windows_size=1000, validation_factor=0.2):

        self.windows_size       = windows_size
        self.batch_size         = batch_size
        self.validation_factor  = validation_factor

        self.dataset = self.normalization(dataset, max_min_file, feature_range)
        print("Нормализация данных выполнена.")

        self.numbs_count, self.characts_count = self.dataset.shape

        # Предварительные размеры валидационной и обучающей выборки выборки
        self.valid_size = math.floor(self.numbs_count * self.validation_factor)
        self.training_size = self.numbs_count - self.valid_size

        # Максимальные размеры порций, на которые можно разбить выборки
        self.count_batch_training_dataset = math.floor(self.training_size / self.batch_size)
        self.count_batch_valid_dataset = math.floor(self.valid_size / self.batch_size)

        # Конечные размеры валидационной и обучающей выборки выборки
        self.training_size = self.count_batch_training_dataset * self.batch_size
        self.valid_size = self.count_batch_valid_dataset * self.batch_size

        # Создаём тренировочную и валидационную выборку
        self.training_dataset = self.dataset[:self.training_size]
        self.training_dataset = tf.convert_to_tensor(self.training_dataset, dtype="float32")
        self.training_dataset = tf.reshape(self.training_dataset,
                                           (self.count_batch_training_dataset,
                                            self.batch_size,
                                            self.windows_size,
                                            self.characts_count))

        self.valid_dataset = self.dataset[self.training_size:self.training_size + self.valid_size]
        self.valid_dataset = tf.convert_to_tensor(self.valid_dataset, dtype="float32")
        self.valid_dataset = tf.reshape(self.valid_dataset,
                                        (self.count_batch_valid_dataset,
                                         self.batch_size,
                                         self.windows_size,
                                         self.characts_count))

        # self.numbs_count, self.characts_count = self.dataset.shape
        #
        # self.windows_size       = windows_size
        # self.batch_size         = batch_size
        # self.validation_factor  = validation_factor
        #
        # # Получаем размеры тренировочной и валидационной выборки
        # self.valid_count = math.floor(self.numbs_count * self.validation_factor / self.windows_size) * self.windows_size
        #
        # # Получаем количество характеристик за вычетом валидационной выборки и количество порций данных
        # self.numbs_count = math.floor(math.floor(self.numbs_count / self.windows_size) / self.batch_size) * \
        #                    self.windows_size * self.batch_size - self.valid_count
        # self.count_batch_training_dataset = math.floor(math.floor(self.numbs_count / self.windows_size) / self.batch_size)
        # self.numbs_count = self.count_batch_training_dataset * self.batch_size * self.windows_size
        #
        # print(self.count_batch_training_dataset)
        # print(self.batch_size)
        # print(self.windows_size)
        # print(self.numbs_count)
        # # Создаём тренировочную и валидационную выборку
        # self.training_dataset = self.dataset[:self.numbs_count]
        # self.training_dataset = tf.convert_to_tensor(self.training_dataset, dtype="float32")
        # self.training_dataset = tf.reshape(self.training_dataset,
        #                                    (self.count_batch_training_dataset,
        #                                     self.batch_size,
        #                                     self.windows_size,
        #                                     self.characts_count))
        #
        # self.count_batch_valid_dataset = math.floor(math.floor(self.valid_count / self.windows_size) / self.batch_size)
        # self.valid_dataset = self.dataset[self.numbs_count: self.numbs_count + self.valid_count-self.windows_size]
        # self.valid_dataset = tf.convert_to_tensor(self.valid_dataset, dtype="float32")
        # self.valid_dataset = tf.reshape(self.valid_dataset,
        #                                 (self.count_batch_valid_dataset,
        #                                  self.batch_size,
        #                                  self.windows_size,
        #                                  self.characts_count))

    @staticmethod
    def normalization(pd_data: pd.DataFrame, max_min_file=None, feature_range=(0, 1), mix_max_from_file=False):
        np_data = pd_data.to_numpy(dtype=np.float)
        row_max, col_max = np_data.shape

        if mix_max_from_file:
            read_min_max = pd.read_csv(max_min_file)
            data_max     = read_min_max.iloc[0].to_numpy(dtype=np.float)
            data_min     = read_min_max.iloc[1].to_numpy(dtype=np.float)
        else:
            data_max = pd_data.max().to_numpy(dtype=np.float)
            data_min = pd_data.min().to_numpy(dtype=np.float)

        cols_name = []
        for col in pd_data:
            cols_name.append(col)

        if (not mix_max_from_file) and (max_min_file is not None):
            pd_ch_name = pd.DataFrame([data_max, data_min], columns=cols_name)
            pd_ch_name.to_csv(max_min_file, index=False)

        min_f, max_f = feature_range
        total_max = row_max * col_max
        pbar = tqdm(total=total_max, desc="Прогресс нормализации")
        for col_idx in range(col_max):
            for row_idx in range(row_max):
                pbar.update(1)
                if data_max[col_idx] == data_min[col_idx]:
                    np_data[row_idx, col_idx] = min_f
                    continue
                elif np_data[row_idx, col_idx] == np.inf:
                    np_data[row_idx, col_idx] = max_f
                else:
                    np_data[row_idx, col_idx] = (np_data[row_idx, col_idx] - data_min[col_idx]) / \
                                                (data_max[col_idx] - data_min[col_idx])
                    np_data[row_idx, col_idx] = np_data[row_idx, col_idx] * (max_f - min_f) + min_f
        pbar.close()

        return np_data

    # def __len__(self):
    #     return math.floor(math.floor(self.numbs_count / self.windows_size) / self.batch_size)
    #
    # def __getitem__(self, idx):
    #     batch_x = tf.gather(self.training_dataset, idx, axis=0)
    #     return batch_x
    #
    # def get_valid_len(self):
    #     return math.floor((self.valid_count - self.windows_size) / self.windows_size)
    #
    # def get_valid(self):
    #     len_valid = self.get_valid_len()
    #     for idx in range(len_valid):
    #         valid_batch_x = tf.gather(self.valid_dataset, idx, axis=0)
    #         yield valid_batch_x
    #
    # def on_epoch_end(self):
    #     print("Перемешивание обучающего датасета")
    #     np.random.shuffle(self.dataset)
    #     # self.dataset = self.dataset[10:]
    #
    #     self.numbs_count, self.characts_count = self.dataset.shape
    #
    #     # Получаем размеры тренировочной и валидационной выборки
    #     self.valid_count = math.floor(self.numbs_count * self.validation_factor / self.windows_size) * self.windows_size
    #
    #     # Получаем количество характеристик за вычетом валидационной выборки и количество порций данных
    #     self.numbs_count = math.floor(math.floor(self.numbs_count / self.windows_size) / self.batch_size) * \
    #                        self.windows_size * self.batch_size - self.valid_count
    #     self.count_batch_training_dataset = math.floor(
    #         math.floor(self.numbs_count / self.windows_size) / self.batch_size)
    #     self.numbs_count = self.count_batch_training_dataset * self.batch_size * self.windows_size
    #
    #     # Создаём тренировочную и валидационную выборку
    #     self.training_dataset = self.dataset[:self.numbs_count]
    #     self.training_dataset = tf.convert_to_tensor(self.training_dataset)
    #     self.training_dataset = tf.reshape(self.training_dataset,
    #                                        (self.count_batch_training_dataset,
    #                                         self.batch_size,
    #                                         self.windows_size,
    #                                         self.characts_count))
    #     print("Перемешивание выполнено успешно")
    def __len__(self):
        return self.count_batch_training_dataset

    def __getitem__(self, idx):
        batch_x = tf.gather(self.training_dataset, idx, axis=0)
        return batch_x

    def get_valid_len(self):
        return self.count_batch_valid_dataset

    def get_valid(self):
        len_valid = self.get_valid_len()
        for idx in range(len_valid):
            valid_batch_x = tf.gather(self.valid_dataset, idx, axis=0)
            yield valid_batch_x

    def on_epoch_end(self):
        print("Перемешивание обучающего датасета")
        self.training_dataset = self.dataset[:self.training_size]
        self.training_dataset = tf.convert_to_tensor(self.training_dataset, dtype="float32")
        self.training_dataset = tf.random.shuffle(self.training_dataset)
        self.training_dataset = tf.reshape(self.training_dataset,
                                           (self.count_batch_training_dataset,
                                            self.batch_size,
                                            self.windows_size,
                                            self.characts_count))
        print(self.training_dataset)
        print("Перемешивание выполнено успешно")

class TrainingDatasetNetFlowTrafficGen(keras.utils.Sequence):
    """
        Класс генератора нормализованных данных для обучения нейронной сети
        автоэнкодера. Позволяет генерировать данные наборами batch_size,
        чтобы в дальнейшем передавать по одному батчу в нейронную сеть для очередного шага обучения.
    """

    def __init__(self, dataset, max_min_file, feature_range,
                 batch_size=1000, windows_size=1000, validation_factor=0.2):

        self.windows_size       = windows_size
        self.batch_size         = batch_size
        self.validation_factor  = validation_factor

        # Подготавливаем данные
        dataset          = self.normalization(dataset, max_min_file, feature_range)
        netflows         = self.split_netflow(dataset)
        self.windows_arr = self.portioning(netflows, self.windows_size)

        self.numbs_count, self.characts_count = dataset.shape
        self.len_windows                      = len(self.windows_arr)
        self.characts_count -= 4

        # Предварительные размеры валидационной и обучающей выборки выборки
        self.valid_size     = math.floor(self.len_windows * self.validation_factor)
        self.training_size  = self.len_windows - self.valid_size

        # Максимальные размеры порций, на которые можно разбить выборки
        self.count_batch_training_dataset = math.floor(self.training_size / self.batch_size)
        self.count_batch_valid_dataset    = math.floor(self.valid_size / self.batch_size)

        # Конечные размеры валидационной и обучающей выборки выборки
        self.training_size = self.count_batch_training_dataset * self.batch_size
        self.valid_size    = self.count_batch_valid_dataset * self.batch_size

        # Создаём тренировочную и валидационную выборку
        self.training_dataset = self.windows_arr[:self.training_size]
        self.training_dataset = tf.convert_to_tensor(self.training_dataset, dtype="float32")
        self.training_dataset = tf.reshape(self.training_dataset,
                                           (self.count_batch_training_dataset,
                                            self.batch_size,
                                            self.windows_size,
                                            self.characts_count))

        self.valid_dataset = self.windows_arr[self.training_size:self.training_size + self.valid_size]
        self.valid_dataset = tf.convert_to_tensor(self.valid_dataset, dtype="float32")
        self.valid_dataset = tf.reshape(self.valid_dataset,
                                        (self.count_batch_valid_dataset,
                                         self.batch_size,
                                         self.windows_size,
                                         self.characts_count))

    @staticmethod
    def normalization(pd_data: pd.DataFrame, max_min_file=None, feature_range=(0, 1), mix_max_from_file=False):
        np_data = pd_data.to_numpy(dtype=np.float)
        row_max, col_max = np_data.shape

        if mix_max_from_file:
            read_min_max = pd.read_csv(max_min_file)
            data_max     = read_min_max.iloc[0].to_numpy(dtype=np.float)
            data_min     = read_min_max.iloc[1].to_numpy(dtype=np.float)
        else:
            data_max = pd_data.max().to_numpy(dtype=np.float)
            data_min = pd_data.min().to_numpy(dtype=np.float)

        cols_name = []
        for col in pd_data:
            cols_name.append(col)

        if (not mix_max_from_file) and (max_min_file is not None):
            pd_ch_name = pd.DataFrame([data_max, data_min], columns=cols_name)
            pd_ch_name.to_csv(max_min_file, index=False)

        min_f, max_f = feature_range
        total_max = row_max * col_max
        pbar = tqdm(total=total_max, desc="Прогресс нормализации")
        for row_idx in range(row_max):
            for col_idx in range(col_max):
                pbar.update(1)
                if col_idx in [0, 1, 2, 3]:
                    continue
                if data_max[col_idx] == data_min[col_idx]:
                    np_data[row_idx, col_idx] = min_f
                    continue
                elif np_data[row_idx, col_idx] == np.inf:
                    np_data[row_idx, col_idx] = max_f
                else:
                    np_data[row_idx, col_idx] = (np_data[row_idx, col_idx] - data_min[col_idx]) / \
                                                (data_max[col_idx] - data_min[col_idx])
                    np_data[row_idx, col_idx] = np_data[row_idx, col_idx] * (max_f - min_f) + min_f
        pbar.close()

        return np_data

    @staticmethod
    def split_netflow(np_data):
        netflows = {}

        row_max, _ = np_data.shape
        pbar = tqdm(total=row_max, desc="Прогресс разделения на потоки")
        for row_idx in range(row_max):
            Src_IP_Flow   = np_data[row_idx, 0]
            Dst_IP_Flow   = np_data[row_idx, 1]
            Src_Port_Flow = np_data[row_idx, 2]
            Dst_Port_Flow = np_data[row_idx, 3]

            netflow = int(Src_IP_Flow + Dst_IP_Flow + Src_Port_Flow + Dst_Port_Flow)
            if not netflow in netflows:
                netflows[netflow] = list()

            netflows[netflow].append(np_data[row_idx, 4:])
            pbar.update(1)
        return netflows

    @staticmethod
    def portioning(netflows, window_size):
        pbar = tqdm(total=len(netflows), desc="Прогресс разделения на окна")
        windows = list()
        for flow in netflows:
            batch = list()
            for ch in netflows[flow]:
                if len(batch) < window_size:
                    batch.append(ch)
                else:
                    windows.append(batch.copy())
                    batch.pop(0)
                    batch.append(ch)
            pbar.update(1)

        pbar = tqdm(total=len(windows), desc="Проверка окон на заданный размер")
        new_windows = list()
        for i in range(len(windows)):
            if len(windows[i]) == window_size:
                new_windows.append(windows[i])
            pbar.update(1)

        return new_windows

    def __len__(self):
        return self.count_batch_training_dataset

    def __getitem__(self, idx):
        batch_x = tf.gather(self.training_dataset, idx, axis=0)
        return batch_x

    def get_valid_len(self):
        return self.count_batch_valid_dataset

    def get_valid(self):
        len_valid = self.get_valid_len()
        for idx in range(len_valid):
            valid_batch_x = tf.gather(self.valid_dataset, idx, axis=0)
            yield valid_batch_x

    def on_epoch_end(self):
        print("Перемешивание обучающего датасета")
        self.training_dataset = self.windows_arr[:self.training_size]
        self.training_dataset = tf.convert_to_tensor(self.training_dataset, dtype="float32")
        self.training_dataset = tf.random.shuffle(self.training_dataset)
        self.training_dataset = tf.reshape(self.training_dataset,
                                           (self.count_batch_training_dataset,
                                            self.batch_size,
                                            self.windows_size,
                                            self.characts_count))
        print(self.training_dataset)
        print("Перемешивание выполнено успешно")