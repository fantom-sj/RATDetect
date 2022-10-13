"""
    Модуль в котором содержится архитектура рекуррентной нейронной сети,
    используемой для детектирования аномалий в сетевом трафике.
"""

import tensorflow as tf
from tensorflow import keras
from keras import Model, Sequential
from keras.layers import GRU
from keras.utils import Progbar

import pandas as pd
import numpy as np
import random

from tensorflow.python.client import device_lib


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
        self.windows_size = windows_size
        self.caracts_count = caracts_count
        self.seed_kernel_init = seed_kernel_init
        self.count_hidden_layers = count_hidden_layers

        self.encoder = Sequential(name="Encoder")
        self.middle = Sequential(name="Middle")
        self.decoder = Sequential(name="Decoder")

        index = 0
        for i in range(0, self.count_hidden_layers * (-1), -1):
            self.encoder.add(
                GRU(
                    units=self.caracts_count + i,
                    activation="tanh",
                    kernel_initializer=keras.initializers.he_uniform(self.seed_kernel_init),
                    return_sequences=True,
                    name="enc_hid_layer_" + str(index),
                    input_shape=(self.windows_size, self.caracts_count + i)
                )
            )
            index += 1

        self.middle.add(
            GRU(
                units=self.caracts_count - index,
                activation="tanh",
                kernel_initializer=keras.initializers.he_uniform(self.seed_kernel_init),
                return_sequences=True,
                name="mid_hid_layer",
                input_shape=(self.windows_size, self.caracts_count - index + 1)
            )
        )

        index = 0
        for i in range((self.count_hidden_layers - 1) * (-1), 1, 1):
            self.decoder.add(
                GRU(
                    units=self.caracts_count + i,
                    activation="tanh",
                    kernel_initializer=keras.initializers.he_uniform(self.seed_kernel_init),
                    return_sequences=True,
                    name="dec_hid_layer_" + str(index),
                    input_shape=(self.windows_size, self.caracts_count + i - 1)
                )
            )
            index += 1

        self.loss_tracker = keras.metrics.Mean(name="loss")
        self.valid_loss_tracker = keras.metrics.Mean(name="valid_loss")
        self.mae_metric = keras.metrics.MeanAbsoluteError(name="mae")
        self.history_loss = {"epoch": [], "step": [], "loss": [], "mean_loss": [], "mae": []}
        self.history_valid = {"epoch": [], "loss": [], "mean_loss": []}

    def train_step(self, x_batch_train):
        with tf.GradientTape() as tape:
            logits = self.__call__(x_batch_train)
            loss_value = keras.losses.kl_divergence(x_batch_train, logits)

        trainable_vars = self.trainable_variables
        gradients = tape.gradient(loss_value, trainable_vars)

        # Обновляем веса
        self.optimizer.apply_gradients(zip(gradients, trainable_vars))

        # Обновляем метрику на обучении.
        loss = float(np.mean(np.array(loss_value)[0]))
        self.loss_tracker.update_state(loss_value)
        self.mae_metric.update_state(x_batch_train, logits)

        return loss

    def education(self, training_dataset, epochs=1, shuffle=True, model_chekname="model"):
        loss = 0
        metrics_names = ["Расхождение", "Средние расхождение", "Средняя абсолютная ошибка"]

        for epoch in range(epochs):
            print("Эпоха {}/{}".format(epoch+1, epochs))

            progress_bar = Progbar(round(training_dataset.numbs_count / self.batch_size)-1,
                                   stateful_metrics=metrics_names)

            # Итерируем по пакетам в датасете.
            for step, x_batch_train in enumerate(training_dataset):
                loss = self.train_step(x_batch_train)

                loss_tracker_res = self.loss_tracker.result()
                mae_metric_res = self.mae_metric.result()

                # Пишем лог после прохождения каждого батча
                self.history_loss["epoch"].append(epoch)
                self.history_loss["step"].append(step)
                self.history_loss["loss"].append(loss)
                self.history_loss["mean_loss"].append(float(loss_tracker_res))
                self.history_loss["mae"].append(float(mae_metric_res))

                values = [("Расхождение", loss),
                          ("Средние расхождение", (float(loss_tracker_res))),
                          ("Средняя абсолютная ошибка", (float(mae_metric_res)))]

                progress_bar.add(1, values=values)

            self.loss_tracker.reset_states()
            self.mae_metric.reset_states()

            valid_metrics_name = ["Расхождение", "Средние расхождение"]
            print("Валидация после эпохи {}".format(epoch+1))
            progress_bar_valid = Progbar(round(training_dataset.valid_count / self.batch_size) - 1,
                                         stateful_metrics=valid_metrics_name)


            for valid_batch_x in training_dataset.get_valid_item():
                val_logits = self.__call__(valid_batch_x)
                valid_loss_value = keras.losses.kl_divergence(valid_batch_x, val_logits)

                valid_loss = float(np.mean(np.array(valid_loss_value)[0]))
                self.valid_loss_tracker.update_state(valid_loss_value)
                valid_loss_tracker_res = float(self.valid_loss_tracker.result())

                values = [("Расхождение", valid_loss),
                          ("Средние расхождение", valid_loss_tracker_res)]

                # Пишем лог после прохождения каждого батча
                self.history_valid["epoch"].append(epoch)
                self.history_valid["loss"].append(loss)
                self.history_valid["mean_loss"].append(valid_loss_tracker_res)

                progress_bar_valid.add(1, values=values)

            self.valid_loss_tracker.reset_states()

            if (epoch+1) != epochs:
                self.save(model_chekname+"_e"+str(epoch+1))

            if shuffle:
                training_dataset.on_epoch_end()
        print("Обучение завершено!\n")

    def call(self, input_features):
        compression = self.encoder(input_features)
        transfer = self.middle(compression)
        reconstruction = self.decoder(transfer)
        return reconstruction

    @property
    def metrics(self):
        return [self.loss_tracker, self.mae_metric]


class TrainingDatasetGen(keras.utils.Sequence):
    """
        Класс генератора нормализованных данных для обучения нейронной сети
        автоэнкодера. Позволяет генерировать данные наборами batch_size,
        чтобы в дальнейшем передавать по одному батчу в нейронную сеть для очередного шага обучения.
    """

    def __init__(self, dataset, max_min_file, batch_size=1000, windows_size=1000, validation_factor=0.2):
        # Нормализуем данные
        self.dataset = self.normalization(dataset, max_min_file).to_numpy()
        print("Нормализация данных выполнена.")

        self.numbs_count, self.caracts_count = self.dataset.shape

        self.windows_size = windows_size
        self.batch_size = batch_size

        # Получаем размеры тренировочной и валидационной выборки
        self.valid_count = round(self.numbs_count * validation_factor / batch_size) * batch_size
        self.numbs_count = round(len(self.dataset) / batch_size) * batch_size - self.valid_count

        # Создаём тренировочную и валидационную выборку
        self.training_dataset = self.dataset[:self.numbs_count]
        self.valid_dataset    = self.dataset[self.numbs_count:round(len(self.dataset) / batch_size) * batch_size]

    @staticmethod
    def normalization(pd_data, max_min_file):
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
        return round((self.numbs_count - self.windows_size) / self.batch_size)

    def __getitem__(self, idx):
        batch_x = np.array([self.training_dataset[idx * self.batch_size:idx * self.batch_size + self.windows_size, :]])
        return batch_x

    def get_valid_len(self):
        return round((self.valid_count - self.windows_size) / self.batch_size)

    def get_valid_item(self):
        len_valid = self.get_valid_len()
        for idx in range(len_valid):
            valid_batch_x = np.array(
                [self.valid_dataset[idx * self.batch_size:idx * self.batch_size + self.windows_size, :]])
            yield valid_batch_x

    def on_epoch_end(self):
        np.random.shuffle(self.training_dataset)
        np.random.shuffle(self.valid_dataset)


def main():
    # Параметры датасета
    batch_size          = 1000
    validation_factor   = 0.15
    windows_size        = 1000

    # Параметры оптимизатора
    init_learning_rate  = 0.1
    decay_steps         = 1500
    decay_rate          = 0.96
    staircase           = True

    # Параметры нейронной сети
    count_hidden_layers = 7
    epochs              = 10
    seed                = random.randint(3654756461, 9834548734)
    shuffle             = True
    model_name          = "model_GRU_traffic_h7"
    max_min_file        = "M&M_traffic.csv"
    dataset             = "C:\\Users\\Admin\\SnHome\\P2\\characts_06.csv"
    history_name        = "History_train_traffic_1.csv"
    history_valid_name  = "History_valid_traffic_1.csv"

    data = pd.read_csv(dataset)
    data = data.drop(["Time_Stamp"], axis=1)
    print("Загрузка датасета завершена.")

    training_dataset = TrainingDatasetGen(data, max_min_file, batch_size, windows_size, validation_factor)
    print(training_dataset.numbs_count, training_dataset.caracts_count)
    print("Обучающий датасет создан.")

    autoencoder = Autoencoder(caracts_count=training_dataset.caracts_count, seed_kernel_init=seed,
                              batch_size=batch_size, windows_size=windows_size,
                              count_hidden_layers=count_hidden_layers)

    lr_schedule = keras.optimizers.schedules.ExponentialDecay(
        init_learning_rate,
        decay_steps=decay_steps,
        decay_rate=decay_rate,
        staircase=staircase)

    optimizer = keras.optimizers.Adam(learning_rate=lr_schedule)

    autoencoder.compile(optimizer=optimizer)
    print("Автоэнкодер определён.")

    print("Начинаем обучение:")
    autoencoder.education(training_dataset, epochs=epochs, shuffle=shuffle, model_chekname=model_name)
    autoencoder.build((batch_size, windows_size, training_dataset.caracts_count))
    autoencoder.summary()
    autoencoder.encoder.summary()
    autoencoder.middle.summary()
    autoencoder.decoder.summary()
    autoencoder.save(model_name)

    pd.DataFrame(autoencoder.history_loss).to_csv(history_name, index=False)
    pd.DataFrame(autoencoder.history_valid).to_csv(history_valid_name, index=False)

if __name__ == '__main__':
    # print(device_lib.list_local_devices())
    main()
