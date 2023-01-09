from AnomalyDetector.AutoEncoder_RNN import TrainingDatasetGen
from AuxiliaryFunctions import GetFilesCSV
from ProcessEventAnalis.AnalysisProcessEvents import AnalyzerEvents
from ProcessEventAnalis.EventsСharacts import Events_Charact
from tensorflow import keras
from keras import Model

from keras.utils import Progbar
from threading import Thread
from pathlib import Path

import tensorflow as tf
import pandas as pd
import numpy as np
import logging
import pickle
import socket
import struct
import math
import time
import copy
import ssl
import sys


def receive_data(connection):
    data_size = struct.unpack('>I', connection.recv(4))[0]
    received_payload = b""
    reamining_payload_size = data_size
    while reamining_payload_size != 0:
        received_payload += connection.recv(reamining_payload_size)
        reamining_payload_size = data_size - len(received_payload)
    return pickle.loads(received_payload)


class EventAnalyser(Thread):
    def __init__(self, buffer: list, NetEvent: Model,
                 protected_devices: dict):
        super().__init__()

        # Параметры анализатора событий
        self.log_file            = "LogStdOut.txt"
        self.err_file            = "LogStdErr.txt"
        self.NetEvent            = NetEvent
        self.protected_devices   = protected_devices
        self.buffer              = buffer

        # Параметры анализа с помощью нейросети
        self.batch_size     = 1
        self.window_size    = 1
        self.loss_func      = keras.losses.mse
        self.max_min_file   = "AnomalyDetector\\modeles\\EventAnomalyDetector\\0.6.0\\M&M_event.csv"
        self.feature_range  = (-1, 1)

        self.context = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
        self.context.load_cert_chain(certfile="certificates\\server.crt", keyfile="certificates\\server.key")

        self.characts           = dict()
        self.threads_collector  = dict()

    def NeiroAnalyze(self, characts):
        characts_new = characts[characts[Events_Charact.Process_Name] != "python.exe"]
        characts_new.sort_values(by=Events_Charact.Time_Stamp_End)

        characts_pd = characts_new.drop([Events_Charact.Time_Stamp_Start], axis=1)
        characts_pd = characts_pd.drop([Events_Charact.Time_Stamp_End], axis=1)
        characts_pd = characts_pd.drop([Events_Charact.Process_Name], axis=1)
        characts_pd = characts_pd.drop([Events_Charact.Direction_IP_Port], axis=1)
        characts_pd = characts_pd.drop([Events_Charact.Count_Events_Batch], axis=1)
        characts_pd = characts_pd.drop([Events_Charact.Duration], axis=1)
        caracts_numpy = TrainingDatasetGen.normalization(characts_pd, self.max_min_file, self.feature_range,
                                                         True)
        characts_dt    = characts_new.to_dict("list")

        numbs_count, caracts_count = caracts_numpy.shape
        batch_count = math.floor(numbs_count / self.batch_size)

        print("Начинаем прогнозирование аномальных событий.")
        metric_loss = None

        # progress_bar = Progbar(batch_count, stateful_metrics=["Расхождение"])

        for idx in range(0, batch_count, 1):
            batch_x = []
            for i in range(self.batch_size):
                batch_x.append(caracts_numpy[i + (idx * self.batch_size):self.window_size + i + (idx * self.batch_size)])
            try:
                batch_x = tf.convert_to_tensor(batch_x)
                batch_x_restored = self.NetEvent.__call__(batch_x)

                loss = self.loss_func(batch_x, batch_x_restored)
                loss = tf.math.reduce_mean(loss, 1)
                if idx == 0:
                    metric_loss = loss
                else:
                    metric_loss = tf.concat([metric_loss, loss], axis=0)
                mean_loss = tf.math.reduce_mean(
                    loss)  # tf.math.multiply(tf.math.reduce_mean(loss), tf.constant(1, dtype=tf.float32))
                values = [("Расхождение", mean_loss)]
                # progress_bar.add(1, values=values)

                self.buffer.append((characts_dt[Events_Charact.Time_Stamp_Start][idx],
                                    characts_dt[Events_Charact.Time_Stamp_End][idx],
                                    characts_dt[Events_Charact.Process_Name][idx],
                                    characts_dt[Events_Charact.Direction_IP_Port][idx],
                                    float(metric_loss[idx])
                                    ))

            except Exception as err:
                logging.exception(f"Ошибка!\n{err}")
                print(np.array(batch_x).shape)
                continue
        print("Анализ с помощью нейросети NetEvent завершён")

    def data_collector(self, name_device, device):
        HOST, PORT = device
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM, 0) as sock:
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)

            sock.bind((HOST, PORT))
            sock.listen(0)

            print(f"Запущен сборщик событий для устройства: {name_device}")
            with self.context.wrap_socket(sock, server_side=True) as ssock:
                while True:
                    connection, client_address = ssock.accept()
                    data = receive_data(connection)
                    connection.close()
                    self.characts[name_device].append(data)

    def run(self):
        with open(self.err_file, "w") as f:         # sys.stderr
            with open(self.log_file, "w") as f:     # sys.stdout
                print("Поток анализа событий процессов запущен!")

                for name_device in self.protected_devices:
                    self.characts[name_device] = list()
                    self.threads_collector[name_device] = Thread(target=self.data_collector, args=(name_device,
                                                                    self.protected_devices[name_device],))
                    self.threads_collector[name_device].start()

                while True:
                    for name_device in self.characts:
                        characts_data = copy.copy(self.characts[name_device])

                        if len(characts_data) > 0:
                            self.characts[name_device].clear()
                            # print(characts_data[0])
                            # print(type(characts_data[0]))
                            self.NeiroAnalyze(characts_data[0])
                    time.sleep(1)