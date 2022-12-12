from AnomalyDetector.AutoEncoder_RNN import TrainingDatasetGen
from AuxiliaryFunctions import GetFilesCSV
from ProcessEventAnalis.AnalysisProcessEvents import AnalyzerEvents
from tensorflow import keras
from keras import Model

from keras.utils import Progbar
from threading import Thread
from pathlib import Path

import tensorflow as tf
import pandas as pd
import numpy as np
import logging
import math
import time
import sys


class EventAnalyser(Thread):
    def __init__(self, buffer: list, NetEvent: Model,
                 path_event_analysis: str):
        super().__init__()

        # Параметры анализатора событий
        self.log_file            = "LogStdOut.txt"
        self.err_file            = "LogStdErr.txt"
        self.NetEvent            = NetEvent
        self.path_event_analysis = path_event_analysis
        self.buffer              = buffer

        # Параметры анализа с помощью нейросети
        self.batch_size = 1
        self.window_size = 1
        self.loss_func = keras.losses.mse
        self.max_min_file = "AnomalyDetector\\modeles\\EventAnomalyDetector\\0.4.7.1_LSTM\\M&M_event.csv"
        self.feature_range = (-1, 1)

    def NeiroAnalyze(self, characts):
        caracts_pd = characts.drop(["Time_Stamp_Start"], axis=1)
        caracts_pd = caracts_pd.drop(["Time_Stamp_End"], axis=1)
        caracts_pd = caracts_pd.drop(["Process_Name"], axis=1)
        caracts_pd = caracts_pd.drop(["Direction_IP_Port"], axis=1)
        caracts_pd = caracts_pd.drop(["Count_Events_Batch"], axis=1)
        if "Count_System_Statistics" in caracts_pd.columns:
            caracts_pd = caracts_pd.drop(["Count_System_Statistics"], axis=1)
        caracts_numpy = TrainingDatasetGen.normalization(caracts_pd, self.max_min_file, self.feature_range,
                                                         True).to_numpy()
        caracts_dt    = characts.to_dict("list")

        if np.isnan(np.sum(caracts_numpy)):
            caracts_numpy = np.nan_to_num(caracts_numpy)

        numbs_count, caracts_count = caracts_numpy.shape
        batch_count = math.ceil(numbs_count / self.batch_size)

        metrics_analiz = {}
        # progress_bar = Progbar(batch_count, stateful_metrics=["Расхождение"])

        caracts_tensor = tf.convert_to_tensor(caracts_numpy)
        caracts_tensor_shape = np.array(caracts_tensor.shape)
        caracts_tensor = tf.reshape(caracts_tensor, (caracts_tensor_shape[0], 1, 1, caracts_tensor_shape[1]))

        Direction_IP_Port_unic = {}
        for idx in range(0, batch_count, 1):
            proc = caracts_dt["Process_Name"][idx]
            if isinstance(caracts_dt["Direction_IP_Port"][idx], str):
                if caracts_dt["Direction_IP_Port"][idx].rfind(";") != -1:
                    Direction_IP_Port = caracts_dt["Direction_IP_Port"][idx].split(";")
                else:
                    Direction_IP_Port = [caracts_dt["Direction_IP_Port"][idx]]
                if not proc in Direction_IP_Port_unic:
                    Direction_IP_Port_unic[proc] = []
                for d in Direction_IP_Port:
                    if not d in Direction_IP_Port_unic[proc]:
                        Direction_IP_Port_unic[proc].append(d)
            elif not math.isnan(caracts_dt["Direction_IP_Port"][idx]):
                if not proc in Direction_IP_Port_unic:
                    Direction_IP_Port_unic[proc] = []

        print("Начинаем анализ с помощью нейросети NetEvent")
        for idx in range(0, batch_count, 1):
            batch_x = tf.gather(caracts_tensor, idx)
            try:
                batch_x_restored = self.NetEvent.__call__(batch_x)

                loss = self.loss_func(batch_x, batch_x_restored)
                loss = tf.math.reduce_mean(loss, 1)
                if idx == 0:
                    metrics_analiz["loss"] = loss
                else:
                    metrics_analiz["loss"] = tf.concat([metrics_analiz["loss"], loss], axis=0)
                mean_loss = tf.math.reduce_mean(loss)
                values = [("Расхождение", mean_loss)]
                # progress_bar.add(1, values=values)

                if caracts_dt["Process_Name"][idx] in Direction_IP_Port_unic:
                    Direction_IP_Port_proc = Direction_IP_Port_unic[caracts_dt["Process_Name"][idx]]
                else:
                    Direction_IP_Port_proc = None

                self.buffer.append({caracts_dt["Process_Name"][idx]: (metrics_analiz["loss"][idx],
                                                                 caracts_dt["Time_Stamp_Start"][idx],
                                                                 caracts_dt["Time_Stamp_End"][idx],
                                                                 Direction_IP_Port_proc)})
            except Exception as err:
                logging.exception(f"Ошибка!\n{err}")
                print(np.array(batch_x).shape)
                continue
        print("Анализ с помощью нейросети NetEvent завершён")

    def run(self):
        with open(self.err_file, "w") as f: # sys.stderr
            with open(self.log_file, "w") as f: # sys.stdout
                print("Поток анализа событий процессов запущен!")

                window_size = 50
                charact_file_name = "events_characters_"
                user_dir = "Жертва"
                path_event_analysis = "\\\\VictimPC\\RATDetect\\WorkDirectory"

                analizator = AnalyzerEvents(window_size, charact_file_name,
                                            path_event_analysis, user_dir)

                analizator.run()

                while True:
                    files_characts = GetFilesCSV(self.path_event_analysis)

                    if len(files_characts) > 0:
                        print(f"Обнаружено {len(files_characts)} файлов с характеристиками процессов.")
                        print("Загружаем данные о событиях")
                        characts_all = None
                        for file in files_characts:
                            if characts_all is None:
                                characts_all = pd.read_csv(file)
                            else:
                                temp = pd.read_csv(file)
                                characts_all = pd.concat([characts_all, temp], ignore_index=True)
                            Path(file).unlink()

                        self.NeiroAnalyze(characts_all)
                    else:
                        print("Ждём данные о событиях процессов")
                        time.sleep(5)