"""
    Модуль в котором содержится архитектура рекуррентной нейронной сети,
    используемой для детектирования аномалий в сетевом трафике.
"""

import tensorflow as tf
from tensorflow import keras
from keras import Model
from keras.layers import GRU, LSTM, ConvLSTM1D
from keras.utils import Progbar

import pandas as pd
import numpy as np


class Autoencoder(Model):
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
                  model_checkname="model", versia="1", path_model="", checkpoint=None):
        loss = 0
        metrics_names = ["Расхождение", "Средние расхождение", "Средняя абсолютная ошибка"]

        if checkpoint is None:
            start = 0
        else:
            start = checkpoint

        for epoch in range(start, epochs, 1):
            print("Эпоха {}/{}".format(epoch + 1, epochs))

            progress_bar = Progbar(len(training_dataset),
                                   stateful_metrics=metrics_names)

            itter = 0
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
                itter += 1

                if itter % 1000 == 0:
                    self.save_weights(model_checkname + "itter_" + str(round(itter / 1000)))
                    history_name = path_model + "history_train_i" + str(round(itter / 1000)) + "_v" + versia + ".csv"
                    history_valid_name = path_model + "history_valid_i" + str(round(itter / 1000)) \
                                         + "_v" + versia + ".csv"
                    pd.DataFrame(self.history_loss).to_csv(history_name, index=False)
                    pd.DataFrame(self.history_valid).to_csv(history_valid_name, index=False)

            self.loss_tracker.reset_states()
            self.mae_metric.reset_states()

            valid_metrics_name = ["Расхождение", "Средние расхождение"]
            print("Валидация после эпохи {}".format(epoch + 1))
            progress_bar_valid = Progbar(training_dataset.get_valid_len(),
                                         stateful_metrics=valid_metrics_name)

            self.save_weights(model_checkname + "epoch_" + str(epoch + 1))
            history_name = path_model + "history_train_e" + str(epoch + 1) + "_v" + versia + ".csv"
            history_valid_name = path_model + "history_valid_e" + str(epoch + 1) + "_v" + versia + ".csv"
            pd.DataFrame(self.history_loss).to_csv(history_name, index=False)
            pd.DataFrame(self.history_valid).to_csv(history_valid_name, index=False)

            try:
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
            except:
                print("Ошибка при валидации!")

            if shuffle and (epoch != (epochs - 1)):
                training_dataset.on_epoch_end()

        print("Обучение завершено!\n")

    def call(self, input_features):
        x = input_features
        # print("\n")
        state = None
        for layer in self.encdec:
            x, state = layer(x)
            # print(f"{layer.name}: {tf.shape(x)}")
        return x

    @property
    def metrics(self):
        return [self.loss_tracker, self.mae_metric]


class TrainingDatasetGen(keras.utils.Sequence):
    """
        Класс генератора нормализованных данных для обучения нейронной сети
        автоэнкодера. Позволяет генерировать данные наборами batch_size,
        чтобы в дальнейшем передавать по одному батчу в нейронную сеть для очередного шага обучения.
    """

    def __init__(self, dataset, max_min_file, feature_range, batch_size=1000, windows_size=1000, validation_factor=0.2):
        # Нормализуем данные
        self.dataset = self.normalization(dataset, max_min_file, feature_range).to_numpy() #[:50000]
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
    def normalization(pd_data, max_min_file=None, feature_range=(0, 1)):
        data_max = pd_data.max()
        data_min = pd_data.min()

        if max_min_file is not None:
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
