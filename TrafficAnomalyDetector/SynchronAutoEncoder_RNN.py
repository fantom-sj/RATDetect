"""
    Модуль в котором содержится архитектура рекуррентной нейронной сети,
    используемой для детектирования аномалий в сетевом трафике.
"""

import tensorflow as tf
from tensorflow import keras
from keras import Model, Sequential
from keras.layers import GRU, Layer, GRUCell
from keras.utils import Progbar

import pandas as pd
import numpy as np
import random
import pylab

from tensorflow.python.client import device_lib


class SynchronGRU(Layer):
    def __init__(self, windows_size: int, caracts_count: int):
        super(SynchronGRU, self).__init__()
        self.windows_size = windows_size
        self.caracts_count = caracts_count

        self.cells = [GRU(self.windows_size, activation="tanh", name="Layer1", input_shape=(self.windows_size, 1))]
        for i in range(1, self.caracts_count, 1):
            self.cells.append(
                GRU(self.windows_size, activation="tanh", name="Layer"+str(i+1), input_shape=(self.windows_size, 2))
            )

        self.y_features = []

    def call(self, input_features: tf.Tensor):
        arr_x_in_windows = []
        for idx in range(self.caracts_count):
            x = tf.gather(input_features, idx, axis=2)
            arr_x_in_windows.append(x)

        y_arr = []
        tensor = tf.reshape(arr_x_in_windows[0], (1, self.windows_size, 1))
        y_arr.append(self.cells[0](tensor))

        for i in range(1, self.caracts_count, 1):
            tensor = tf.concat([arr_x_in_windows[1], y_arr[i-1]], 0)
            tensor = tf.reshape(tensor, (1, 2, self.windows_size))
            y_arr.append(self.cells[i](tensor))

        y_res = y_arr[0]
        for i in range(1, self.caracts_count, 1):
            y_res = tf.concat([y_res, y_arr[i]], 0)

        y_res = tf.reshape(y_res, (1, self.windows_size, self.caracts_count))
        return y_res


class SynchronAutoencoder(Model):
    """
        Класс описывающий из чего состоит синхронный автоэнкодер и как
        происходит его обучение на каждом шаге, какие вычисляются метрики
        и как происходит обратное распространение ошибки для обучения.
    """

    def __init__(self, caracts_count:int, batch_size=1, windows_size=1000, count_hidden_layers=1):
        super(SynchronAutoencoder, self).__init__()
        self.batch_size = batch_size
        self.windows_size = windows_size
        self.caracts_count = caracts_count
        self.count_hidden_layers = count_hidden_layers

        self.L1  = SynchronGRU(self.windows_size, self.caracts_count)
        self.L2  = SynchronGRU(self.windows_size, self.caracts_count)
        self.L3  = SynchronGRU(self.windows_size, self.caracts_count)

        self.loss_tracker = keras.metrics.Mean(name="loss")
        self.mae_metric = keras.metrics.MeanAbsoluteError(name="mae")

        self.valid_loss_tracker = keras.metrics.Mean(name="valid_loss")
        self.valid_mae_metric = keras.metrics.MeanAbsoluteError(name="valid_mae")

        self.history_loss = {"epoch": [], "step": [], "loss": [], "mean_loss": [], "mae": []}
        self.history_valid = {"epoch": [], "step": [], "loss": [], "mean_loss": [], "mae": []}

    def train_step(self, x_batch_train):
        with tf.GradientTape() as tape:
            logits = self.__call__(x_batch_train)
            loss_value = self.loss(x_batch_train, logits)

        trainable_vars = self.trainable_variables
        gradients = tape.gradient(loss_value, trainable_vars)

        # Обновляем веса
        self.optimizer.apply_gradients(zip(gradients, trainable_vars))

        # Обновляем метрику на обучении.
        loss = float(np.mean(np.array(loss_value)[0]))
        self.loss_tracker.update_state(loss_value)
        self.mae_metric.update_state(x_batch_train, logits)

        return loss

    def education(self, training_dataset, epochs=1, shuffle=True,
                  model_chekname="model", versia="1", path_model=""):
        loss = 0
        metrics_names = ["Расхождение", "Средние расхождение", "Средняя абсолютная ошибка"]

        for epoch in range(epochs):
            print("Эпоха {}/{}".format(epoch+1, epochs))

            progress_bar = Progbar(round(training_dataset.numbs_count / self.batch_size)-1,
                                   stateful_metrics=metrics_names)

            # Итерируем по пакетам в датасете.
            for step, x_batch_train in enumerate(training_dataset):
                loss = self.train_step(x_batch_train) * 100
                loss_tracker_res = self.loss_tracker.result() * 100
                mae_metric_res = self.mae_metric.result() * 100

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

            for step, valid_batch_x in enumerate(training_dataset.get_valid()):
                val_logits = self.__call__(valid_batch_x)
                valid_loss_value = self.loss(valid_batch_x, val_logits)

                self.valid_loss_tracker.update_state(valid_loss_value)
                self.valid_mae_metric.update_state(valid_batch_x, val_logits)

                valid_loss = float(np.mean(np.array(valid_loss_value)[0])) * 100
                valid_loss_tracker_res = float(self.valid_loss_tracker.result()) * 100
                valid_mae_metric_res = float(self.valid_mae_metric.result()) * 100

                values = [("Расхождение", valid_loss),
                          ("Средние расхождение", valid_loss_tracker_res),
                          ("Средняя абсолютная ошибка", valid_mae_metric_res)]

                # Пишем лог после прохождения каждого батча
                self.history_valid["epoch"].append(epoch)
                self.history_valid["step"].append(step)
                self.history_valid["loss"].append(loss)
                self.history_valid["mean_loss"].append(valid_loss_tracker_res)
                self.history_valid["mae"].append(valid_loss_tracker_res)

                progress_bar_valid.add(1, values=values)

            self.valid_loss_tracker.reset_states()
            self.valid_mae_metric.reset_state()

            if (epoch+1) != epochs:
                self.save(model_chekname+"_e"+str(epoch+1))
                history_name = path_model + "history_train_e" + str(epoch+1) + "_v" + versia + ".csv"
                history_valid_name = path_model + "history_valid_e" + str(epoch+1) + "_v" + versia + ".csv"
                pd.DataFrame(self.history_loss).to_csv(history_name, index=False)
                pd.DataFrame(self.history_valid).to_csv(history_valid_name, index=False)

            if shuffle:
                training_dataset.on_epoch_end()

        print("Обучение завершено!\n")

    def call(self, input_features):
        res = self.L1(input_features)
        res = self.L2(res)
        res = self.L3(res)
        return res

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
        self.dataset = self.normalization(dataset, max_min_file).to_numpy() # [:5000]
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
        batch_x = tf.convert_to_tensor(batch_x)
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
        np.random.shuffle(self.training_dataset)
        np.random.shuffle(self.valid_dataset)


def main():
    # Параметры датасета
    batch_size          = 500
    validation_factor   = 0.05
    windows_size        = 500

    # Параметры оптимизатора
    init_learning_rate  = 0.1
    decay_steps         = 1500
    decay_rate          = 0.96
    staircase           = True

    # Параметры нейронной сети
    count_hidden_layers = 3
    epochs              = 2
    shuffle             = True
    loss_func           = keras.losses.mse
    path_model          = "modeles\\SunchronTrafficAnomalyDetector\\"
    versia              = "0.1"
    model_name          = path_model + "model_STAD_v" + versia
    max_min_file        = path_model + "M&M_traffic_VNAT.csv"
    dataset             = "F:\\VNAT\\VNAT_nonvpn_and_characts_06.csv"
    history_name        = path_model + "history_train_v" + versia + ".csv"
    history_valid_name  = path_model + "history_valid_v" + versia + ".csv"

    data = pd.read_csv(dataset, )
    data = data.drop(["Time_Stamp"], axis=1)
    print("Загрузка датасета завершена.")

    training_dataset = TrainingDatasetGen(data, max_min_file, batch_size, windows_size, validation_factor)
    print(training_dataset.numbs_count, training_dataset.caracts_count)
    print("Обучающий датасет создан.")

    autoencoder = SynchronAutoencoder(training_dataset.caracts_count, batch_size, windows_size, count_hidden_layers)

    lr_schedule = keras.optimizers.schedules.ExponentialDecay(
        init_learning_rate,
        decay_steps=decay_steps,
        decay_rate=decay_rate,
        staircase=staircase
    )

    optimizer = keras.optimizers.Adam(learning_rate=lr_schedule)

    autoencoder.compile(optimizer="adam", loss=loss_func)
    print("Автоэнкодер определён.")

    print("Начинаем обучение:")
    autoencoder.education(training_dataset, epochs=epochs, shuffle=shuffle,
                          model_chekname=model_name, versia=versia,
                          path_model=path_model)
    autoencoder.build((batch_size, windows_size, training_dataset.caracts_count))
    autoencoder.summary()
    autoencoder.encoder.summary()
    autoencoder.middle.summary()
    autoencoder.decoder.summary()
    autoencoder.save(model_name)

    pylab.subplot(1, 3, 1)
    pylab.plot(autoencoder.history_loss["loss"])
    pylab.title("Расхождение")

    pylab.subplot(1, 3, 1)
    pylab.plot(autoencoder.history_loss["mean_loss"])
    pylab.title("Средние расхождение")

    pylab.subplot(1, 3, 1)
    pylab.plot(autoencoder.history_loss["mae"])
    pylab.title("Средняя абсолютная ошибка")

    pd.DataFrame(autoencoder.history_loss).to_csv(history_name, index=False)
    pd.DataFrame(autoencoder.history_valid).to_csv(history_valid_name, index=False)


if __name__ == '__main__':
    # print(device_lib.list_local_devices())

    # batch_size = 1
    # windows_size = 4
    # caracts_count = 3
    #
    # x = [[[0.1, 0.2, 0.3],
    #       [0.4, 0.5, 0.6],
    #       [0.7, 0.8, 0.9],
    #       [0.01, 0.11, 0.12]]]
    # x = tf.convert_to_tensor(x)
    # print(x)
    #
    # autoencoder = SynchronAutoencoder(caracts_count, batch_size, windows_size, 1)
    # y = autoencoder(x)
    # print(y)

    gpu_devices = tf.config.experimental.list_physical_devices("GPU")
    for device in gpu_devices:
        tf.config.experimental.set_memory_growth(device, True)
    tf.config.experimental.set_virtual_device_configuration(gpu_devices[0], [
        tf.config.experimental.VirtualDeviceConfiguration(memory_limit=6000)])

    main()
