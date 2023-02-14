from AnomalyDetector.AutoEncoder_RNN import TrainingDatasetGen
from ProcessEventAnalis.EventsСharacts import Events_Charact, OperationName

from keras.utils import Progbar
from tensorflow import keras
from keras import Model

from threading import Thread

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

        self.run_toggle                = False
        self.run_toggle_data_collector = False

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
        self.max_min_file   = "AnomalyDetector\\modeles\\EventAnomalyDetector\\0.7.0\\M&M_event.csv"
        self.feature_range  = (-1, 1)

        self.context = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
        self.context.load_cert_chain(certfile="certificates\\server.crt", keyfile="certificates\\server.key")

        self.characts           = dict()
        self.threads_collector  = dict()

    def NeiroAnalyze(self, characts: pd.DataFrame):
        characts_new = characts[characts[Events_Charact.Process_Name] != "python.exe"]
        characts_new.sort_values(by=Events_Charact.Time_Stamp_End)

        characts_pd = characts_new.drop([Events_Charact.Time_Stamp_Start], axis=1)
        characts_pd = characts_pd.drop([Events_Charact.Time_Stamp_End], axis=1)
        characts_pd = characts_pd.drop([Events_Charact.Process_Name], axis=1)
        characts_pd = characts_pd.drop([Events_Charact.Direction_IP_Port], axis=1)
        characts_pd = characts_pd.drop([Events_Charact.Count_Events_Batch], axis=1)
        characts_pd = characts_pd.drop([Events_Charact.Duration], axis=1)

        # Выявленные ненужные признаки:
        data = characts_pd.drop([Events_Charact.Count_Process_Defined], axis=1)
        data = data.drop([Events_Charact.Count_Thread_Profile], axis=1)
        data = data.drop([Events_Charact.Ratio_Receive_on_Accept], axis=1)
        data = data.drop([Events_Charact.Ratio_Send_on_Accept], axis=1)
        data = data.drop([OperationName.Accept], axis=1)
        data = data.drop([OperationName.CreateMailSlot], axis=1)
        data = data.drop([OperationName.CreatePipe], axis=1)
        data = data.drop([OperationName.DeviceChange], axis=1)
        data = data.drop([OperationName.DirectoryControl], axis=1)
        data = data.drop([OperationName.InternalDeviceIoControl], axis=1)
        data = data.drop([OperationName.LockUnlockFile], axis=1)
        data = data.drop([OperationName.PlugAndPlay], axis=1)
        data = data.drop([OperationName.QueryFileQuota], axis=1)
        data = data.drop([OperationName.QueryInformationFile], axis=1)
        data = data.drop([OperationName.RenameKey], axis=1)
        data = data.drop([OperationName.SetFileQuota], axis=1)
        data = data.drop([OperationName.SetInformationFile], axis=1)
        data = data.drop([OperationName.SetVolumeInformation], axis=1)
        data = data.drop([OperationName.VolumeDismount], axis=1)
        data = data.drop([OperationName.VolumeMount], axis=1)
        data = data.drop([Events_Charact.Appeal_reg_HKCC], axis=1)
        data = data.drop([Events_Charact.Speed_Read_Data], axis=1)
        data = data.drop([Events_Charact.Speed_Write_Data], axis=1)
        data = data.drop([OperationName.FlushKey], axis=1)
        data = data.drop([OperationName.LoadKey], axis=1)
        data = data.drop([OperationName.QueryVolumeInformation], axis=1)
        data = data.drop([OperationName.SetEAFile], axis=1)
        data = data.drop([OperationName.SetKeySecurity], axis=1)
        data = data.drop([OperationName.UnloadKey], axis=1)
        data = data.drop([OperationName.SystemControl], axis=1)

        caracts_numpy = TrainingDatasetGen.normalization(data, self.max_min_file, self.feature_range,
                                                         True)
        characts_dt    = characts_new.to_dict("list")

        numbs_count, caracts_count = caracts_numpy.shape
        batch_count = math.floor(numbs_count / self.batch_size)

        print("Начинаем прогнозирование аномальных событий.")
        metric_loss = None

        progress_bar = Progbar(batch_count, stateful_metrics=["Расхождение"])

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
                progress_bar.add(1, values=values)

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
                while self.run_toggle_data_collector:
                    connection, client_address = ssock.accept()
                    data = receive_data(connection)
                    print(f"Приняты данные о событиях от устройства: {name_device}")
                    connection.close()
                    self.characts[name_device].append(data)
                print(f"Сборщик событий для устройства: {name_device} остановлен!")
                return

    def run(self):
        with open(self.err_file, "w") as f:         # sys.stderr
            with open(self.log_file, "w") as f:     # sys.stdout
                print("Поток анализа событий процессов запущен!")

                # events_pd = pd.read_csv("D:\\Пользователи\\Admin\\Рабочий стол\\"
                #                         "Статья по КБ\\RATDetect\\WorkDirectory\\events_characts.csv")
                # self.NeiroAnalyze(events_pd)
                #
                # while True:
                #     time.sleep(1)

                self.run_toggle_data_collector = True
                for name_device in self.protected_devices:
                    self.characts[name_device] = list()
                    self.threads_collector[name_device] = Thread(target=self.data_collector, args=(name_device,
                                                                    self.protected_devices[name_device],))
                    self.threads_collector[name_device].start()

                # file_ch = "WorkDirectory\\events_characts_"
                # idx     = 0
                while self.run_toggle:
                    for name_device in self.characts:
                        characts_data = copy.copy(self.characts[name_device])

                        if len(characts_data) > 0:
                            self.characts[name_device].clear()

                            # file_name = file_ch + str(idx) + ".csv"
                            # characts_data[0].to_csv(file_name, index=False)
                            # idx += 1
                            # print(f"Файл: {file_name} сохранён")

                            # print(characts_data[0])
                            # print(type(characts_data[0]))
                            try:
                                self.NeiroAnalyze(characts_data[0])
                            except Exception as err:
                                logging.exception(f"\nНе верный формат полученных данных\n{err}")
                    time.sleep(2)

                self.run_toggle_data_collector = False
                print("Поток анализа событий процессов остановлен!")
                return