"""
    Модуль в котором содержится архитектура рекуррентной нейронной сети,
    используемой для детектирования аномалий в сетевом трафике.
"""
from abc import ABCMeta
from keras.utils import Progbar
from tensorflow import keras
from keras import Model
from keras.backend import random_normal, exp, reshape, sum, square

import tensorflow as tf
import numpy as np


def noiser(args):
    z_mean, z_log_var, batch_size, hidden_space = args
    N = random_normal(shape=(batch_size, hidden_space), mean=0, stddev=1.0)
    return exp(z_log_var / 2) * N + z_mean


def loss_for_vae(x, y, args):
    z_mean, z_log_var, batch_size, characts_count = args
    x = reshape(x, shape=(batch_size, characts_count))
    y = reshape(y, shape=(batch_size, characts_count))
    loss = sum(square(x - y), axis=-1)
    kl_loss = -0.5 * sum(1 + z_log_var - square(z_mean) - exp(z_log_var), axis=-1)
    return loss + kl_loss


class AutoencoderBase(Model, metaclass=ABCMeta):
    """
        Класс описывающий из чего состоит автоэнкодер и как
        происходит его обучение на каждом шаге, какие вычисляются метрики
        и как происходит обратное распространение ошибки для обучения.
    """

    def __init__(self):
        super(AutoencoderBase, self).__init__()

        self.loss_tracker = keras.metrics.Mean(name="loss")
        self.mae_metric = keras.metrics.MeanAbsoluteError(name="mae")

        self.valid_loss_tracker = keras.metrics.Mean(name="valid_loss")
        self.valid_mae_metric = keras.metrics.MeanAbsoluteError(name="valid_mae")

        self.characts_count             = None
        self.batch_size                 = None

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

    def education(self, training_dataset, epochs=1, shaffle=False,
                  model_checkname="model", versia="1", path_model="", checkpoint=None):
        metrics_names = ["Расхождение", "Средние расхождение", "Средняя абсолютная ошибка", "Скорость обучения"]

        path_log = "F:\\logdir\\%s"%versia
        summary_writer = tf.summary.create_file_writer(path_log)

        if checkpoint is None:
            start = 0
            checkpoint = 1
        else:
            start = checkpoint

        with summary_writer.as_default():
            global_step = (len(training_dataset)+training_dataset.get_valid_len()) * checkpoint
            tf.summary.trace_on(graph=True)
            # tf.profiler.experimental.Profile(path_log)
            # tf.profiler.experimental.start(path_log)

            for epoch in range(start, epochs, 1):
                print("Эпоха {}/{}".format(epoch + 1, epochs))

                progress_bar = Progbar(len(training_dataset),
                                       stateful_metrics=metrics_names)

                itter = len(training_dataset)*epoch

                for step, x_batch_train in enumerate(training_dataset):
                    with tf.profiler.experimental.Trace("Train", step_num=step):
                        loss = self.train_step(x_batch_train) * 100
                    loss_tracker_res = self.loss_tracker.result() * 100
                    mae_metric_res = self.mae_metric.result() * 100

                    # Пишем лог после прохождения каждого батча
                    global_step += 1
                    learning_rate_value = self.optimizer._decayed_lr(tf.float32)
                    values = [("Ошибка", loss),
                              ("Средние ошибка", (float(loss_tracker_res))),
                              ("Средняя абсолютная ошибка", (float(mae_metric_res))),
                              ("Скорость обучения", learning_rate_value)]

                    tf.summary.scalar("Ошибка", loss, step=global_step)
                    tf.summary.scalar("Средние ошибка", (float(loss_tracker_res)), step=global_step)
                    tf.summary.scalar("Средняя абсолютная ошибка", (float(mae_metric_res)), step=global_step)
                    tf.summary.scalar("Скорость обучения", learning_rate_value, step=global_step)

                    self.history_loss["epoch"].append(epoch)
                    self.history_loss["step"].append(global_step)
                    self.history_loss["loss"].append(float(loss / 100))
                    self.history_loss["mean_loss"].append(float(loss_tracker_res / 100))
                    self.history_loss["mae"].append(float(mae_metric_res / 100))

                    progress_bar.add(1, values=values)
                    itter += 1

                    if itter % 1000 == 0:
                        self.save_weights(model_checkname + "on_itter\\" + "itter_" + str(round(itter / 1000)))

                self.loss_tracker.reset_states()
                self.mae_metric.reset_states()
                try:
                    self.save_weights(model_checkname + "epoch_" + str(epoch + 1))
                    self.save(model_checkname + "epoch_" + str(epoch + 1))
                except Exception as err:
                    print("Ошибка сохранения модели!")
                    print(err)
                
                valid_metrics_name = ["Расхождение", "Средние расхождение"]
                print("Валидация после эпохи {}".format(epoch + 1))
                progress_bar_valid = Progbar(training_dataset.get_valid_len(),
                                             stateful_metrics=valid_metrics_name)

                try:
                    for step, valid_batch_x in enumerate(training_dataset.get_valid()):
                        val_logits = self.__call__(valid_batch_x)
                        valid_loss_value = self.loss(valid_batch_x, val_logits)

                        self.valid_loss_tracker.update_state(valid_loss_value)
                        self.valid_mae_metric.update_state(valid_batch_x, val_logits)

                        valid_loss = float(np.mean(np.array(valid_loss_value)[0])) * 100
                        valid_loss_tracker_res = float(self.valid_loss_tracker.result()) * 100
                        valid_mae_metric_res = float(self.valid_mae_metric.result()) * 100

                        values = [("Ошибка", valid_loss),
                                  ("Средние ошибка", valid_loss_tracker_res),
                                  ("Средняя абсолютная ошибка", valid_mae_metric_res)]

                        # Пишем лог после прохождения каждого батча
                        global_step += 1
                        tf.summary.scalar("Ошибка валидации", valid_loss, step=global_step)
                        tf.summary.scalar("Средние ошибка валидации", valid_loss_tracker_res, step=global_step)
                        tf.summary.scalar("Средняя абсолютная ошибка валидации", valid_mae_metric_res, step=global_step)

                        self.history_valid["epoch"].append(epoch)
                        self.history_valid["step"].append(global_step)
                        self.history_valid["loss"].append(float(valid_loss / 100))
                        self.history_valid["mean_loss"].append(float(valid_loss_tracker_res / 100))
                        self.history_valid["mae"].append(float(valid_mae_metric_res / 100))

                        progress_bar_valid.add(1, values=values)

                    self.valid_loss_tracker.reset_states()
                    self.valid_mae_metric.reset_state()
                except:
                    print("Ошибка при валидации!")

                if shaffle and (epoch != (epochs - 1)):
                    training_dataset.on_epoch_end()

            tf.summary.trace_export("graph", step=global_step, profiler_outdir=path_log)
            # tf.profiler.experimental.stop()

        # tf.summary.trace_off()
        # summary_writer.close()
        print("Обучение завершено!\n")