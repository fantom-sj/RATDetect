from NetTrafficAnalis.StreamingTrafficAnalyzer import SnifferTraffic, AnalyzerPackets
from AnomalyDetector.AutoEncoder_RNN import TrainingDatasetGen
from AuxiliaryFunctions import GetFilesCSV

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


class TrafficAnalyser(Thread):
    def __init__(self, buffer: list, NetTraffic: Model, path_traffic_analysis: str,
                 iface_name: str, ip_client: list, pcap_length=2000, window_size=1000):
        super().__init__()

        # Параметры анализатора трафика
        self.log_file              = "LogStdOut.txt"
        self.err_file              = "LogStdErr.txt"
        self.NetTraffic            = NetTraffic
        self.path_traffic_analysis = path_traffic_analysis
        self.buffer                = buffer

        # Параметры сборщика трафика
        self.pcap_length           = pcap_length
        self.iface_name            = iface_name
        self.trffic_file_mask      = "traffic_"
        self.path_tshark           = "NetTrafficAnalis\\Wireshark\\tshark.exe"

        # Параметры предварительного анализатора трафика
        self.window_size           = window_size
        self.charact_file_length   = 10000
        self.charact_file_name     = "traffic_characters_"
        self.ip_client             = ip_client

        # Параметры нейросетевого анализа
        self.batch_size = 100
        self.loss_func = keras.losses.mse
        self.max_min_file = "AnomalyDetector\\modeles\\TrafficAnomalyDetector\\0.8.7.1\\M&M_traffic_VNAT.csv"
        self.feature_range = (-1, 1)

    def NeiroAnalyze(self, characts):
        caracts_pd = characts.drop(["Time_Stamp_Start"], axis=1)
        caracts_pd = caracts_pd.drop(["Time_Stamp_End"], axis=1)
        caracts_pd = caracts_pd.drop(["Direction_IP_Port"], axis=1)
        caracts_pd = caracts_pd.drop(["Count_src_is_dst_ports"], axis=1)
        caracts_pd = caracts_pd.drop(["Dev_size_TCP_paket"], axis=1)
        caracts_pd = caracts_pd.drop(["Dev_size_UDP_paket"], axis=1)
        caracts_pd = caracts_pd.drop(["Dev_client_paket_size"], axis=1)
        caracts_pd = caracts_pd.drop(["Dev_server_paket_size"], axis=1)
        caracts_numpy = TrainingDatasetGen.normalization(caracts_pd,
                                                         self.max_min_file, self.feature_range, True).to_numpy()

        caracts_dt = characts.to_dict("list")

        numbs_count, caracts_count    = caracts_numpy.shape
        batch_count                   = math.floor(numbs_count/self.batch_size) - 10

        print("Начинаем прогнозирование аномального трафика.")
        metric_loss = None

        valid_metrics_name = ["Расхождение"]
        # progress_bar = Progbar(batch_count, stateful_metrics=valid_metrics_name)

        print("Начинаем анализ с помощью нейросети NetTraffic")
        for idx in range(0, batch_count, 1):
            batch_x = []

            for i in range(self.batch_size):
                batch_x.append(caracts_numpy[i + (idx * self.batch_size):
                                             self.window_size + i + (idx * self.batch_size)])
            try:
                batch_x = tf.convert_to_tensor(batch_x)
                batch_x_restored = self.NetTraffic.predict(batch_x, verbose=0)

                loss = self.loss_func(batch_x, batch_x_restored)
                loss = tf.math.reduce_mean(loss, 1)
                if metric_loss is None:
                    metric_loss = loss
                else:
                    metric_loss = tf.concat([metric_loss, loss], axis=0)
                mean_loss = tf.math.reduce_mean(loss)
                values = [("Расхождение", mean_loss)]
                # progress_bar.add(1, values=values)

                for i in range(self.batch_size):
                    Direction_IP_Port_batch = caracts_dt["Direction_IP_Port"][i + (idx * self.batch_size):
                                                                              self.window_size + i + (
                                                                                          idx * self.batch_size)]

                    Direction_IP_Port_unic = []
                    for d in Direction_IP_Port_batch:
                        d_arr = d.split(";")
                        for one_d in d_arr:
                            if not one_d in Direction_IP_Port_unic:
                                Direction_IP_Port_unic.append(one_d)

                    self.buffer.append({caracts_dt["Time_Stamp_Start"][i + (idx * self.batch_size)]:
                                        (metric_loss[i + (idx * self.batch_size)],
                                         caracts_dt["Time_Stamp_End"]
                                         [self.window_size + i + (idx * self.batch_size)],
                                         Direction_IP_Port_unic)})

            except Exception as err:
                logging.exception(f"Ошибка!\n{err}")
                print(np.array(batch_x).shape)
                continue
        print("Анализ с помощью нейросети NetTraffic завершён")

    def run(self):
        with open(self.err_file, "w") as f:  # sys.stderr
            with open(self.log_file, "w") as f:  # sys.stdout

                print("Поток анализа сетевого трафика запущен!")

                # sniffer = SnifferTraffic(self.pcap_length, self.iface_name, self.path_tshark,
                #                          self.trffic_file_mask, self.path_traffic_analysis)
                # sniffer.run()

                analizator = AnalyzerPackets(self.window_size, self.charact_file_length, self.charact_file_name,
                                             self.ip_client, self.path_traffic_analysis)
                analizator.run()

                while True:
                    files_characts = GetFilesCSV(self.path_traffic_analysis)

                    if len(files_characts) > 0:
                        print(f"Обнаружено {len(files_characts)} файлов с характеристиками трафика.")
                        print("Загружаем данные о трафике")

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
                        print("Ждём данные о трафике")
                        time.sleep(5)


# max_index = idx * self.batch_size + self.batch_size if idx != (batch_count - 1) else numbs_count
# indexis = [range(idx * self.batch_size, max_index, 1)]
