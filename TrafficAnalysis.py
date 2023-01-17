from NetTrafficAnalis.StreamingTrafficAnalyzer import SnifferTraffic, AnalyzerPackets, HUNDREDS_OF_NANOSECONDS
from AnomalyDetector.AutoEncoder_RNN import Autoencoder
from AuxiliaryFunctions import GetFilesCSV

from tensorflow import keras
from keras import Model

from keras.utils import Progbar
from threading import Thread
from pathlib import Path
from enum import Enum
import matplotlib.pyplot as plt

import tensorflow as tf
tf.config.set_visible_devices([], 'GPU')
import pandas as pd
import numpy as np
import logging
import math
import time
import sys


class NetFlowHead(Enum):
    Time_Stamp_Start = 0
    Time_Stamp_End   = 1
    Src_IP_Flow      = 2
    Dst_IP_Flow      = 3
    Src_Port_Flow    = 4
    Dst_Port_Flow    = 5
    Sum_Name_Flow    = 6


class NetFlowGen:
    def __init__(self, max_min_file, feature_range, batch_size=1, windows_size=10, characts_count=51):
        self.windows_size   = windows_size
        self.batch_size     = batch_size
        self.characts_count = characts_count
        self.feature_range  = feature_range

        read_min_max = pd.read_csv(max_min_file)
        self.data_max = read_min_max.iloc[0].to_numpy(dtype=np.float32)
        self.data_min = read_min_max.iloc[1].to_numpy(dtype=np.float32)

        self.NetFlows = dict()

    def appendTraffic(self, traffic: pd.DataFrame):
        traffic_sort = traffic.sort_values(by="Flow_Charact.Time_Stamp_Start")
        traffic_dict = traffic_sort.loc[:, "Flow_Charact.Time_Stamp_Start":
                                           "Flow_Charact.Dst_Port_Flow"].to_dict("records")
        traffic_nump = np.delete(np.array(traffic_sort, dtype=np.float32), [i for i in range(6)], axis=1)
        traffic_norm = self.normalization(traffic_nump, self.data_max, self.data_min, self.feature_range)

        for idx in range(len(traffic_nump)):
            flow = traffic_dict[idx]
            flow_head = dict()
            flow_head[NetFlowHead.Time_Stamp_Start] = flow["Flow_Charact.Time_Stamp_Start"]
            flow_head[NetFlowHead.Time_Stamp_End] = flow["Flow_Charact.Time_Stamp_End"]
            flow_head[NetFlowHead.Src_IP_Flow]    = flow["Flow_Charact.Src_IP_Flow"]
            flow_head[NetFlowHead.Dst_IP_Flow]    = flow["Flow_Charact.Dst_IP_Flow"]
            flow_head[NetFlowHead.Src_Port_Flow]  = flow["Flow_Charact.Src_Port_Flow"]
            flow_head[NetFlowHead.Dst_Port_Flow]  = flow["Flow_Charact.Dst_Port_Flow"]
            Sum_Name_Flow  = sum([flow["Flow_Charact.Src_IP_Flow"],
                                  flow["Flow_Charact.Dst_IP_Flow"],
                                  flow["Flow_Charact.Src_Port_Flow"],
                                  flow["Flow_Charact.Dst_Port_Flow"]])
            dataflow = traffic_norm[idx]

            if not Sum_Name_Flow in self.NetFlows:
                self.NetFlows[Sum_Name_Flow] = list()

            self.NetFlows[Sum_Name_Flow].append((flow_head, dataflow))

    @staticmethod
    def normalization(np_data: np.array, data_max, data_min, feature_range=(-1, 1)):
        row_max, col_max = np_data.shape
        norm_data_np     = np.zeros((row_max, col_max))

        # data_max = list()
        # data_min = list()
        #
        # for data in np.transpose(np_data):
        #     data_max.append(np.max(data))
        #     data_min.append(np.min(data))

        min_f, max_f = feature_range
        for row_idx in range(row_max):
            for col_idx in range(col_max):
                if data_max[col_idx] == data_min[col_idx]:
                    norm_data_np[row_idx, col_idx] = min_f
                    continue
                elif np_data[row_idx, col_idx] == np.inf:
                    norm_data_np[row_idx, col_idx] = max_f
                else:
                    norm_data_np[row_idx, col_idx] = (np_data[row_idx, col_idx] - data_min[col_idx]) / \
                                                (data_max[col_idx] - data_min[col_idx])
                    norm_data_np[row_idx, col_idx] = np_data[row_idx, col_idx] * (max_f - min_f) + min_f

        return norm_data_np

    def __iter__(self):
        for flow_name in self.NetFlows:
            batch_flow_head = list()
            batch_dataflow  = list()
            for flow in self.NetFlows[flow_name]:
                flow_head, dataflow = flow
                if len(batch_flow_head) < self.windows_size:
                    batch_flow_head.append(flow_head)
                    batch_dataflow.append(dataflow)
                else:
                    batch_dataflow_tf = tf.convert_to_tensor(batch_dataflow, dtype="float32")
                    batch_dataflow_tf = tf.reshape(batch_dataflow_tf,
                                                   (self.batch_size,
                                                    self.windows_size,
                                                    self.characts_count))

                    yield batch_flow_head.copy(), batch_dataflow_tf
                    batch_flow_head.pop(0)
                    batch_dataflow.pop(0)
                    batch_flow_head.append(flow_head)
                    batch_dataflow.append(dataflow)


class TrafficAnalyser(Thread):
    def __init__(self, buffer: list, NetTraffic: Model, path_traffic_analysis: str,
                 iface_name: str, ip_client: list, pcap_length=2000, window_size=10):
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
        self.flow_time_limit       = 1 * 60 * HUNDREDS_OF_NANOSECONDS
        self.traffic_waiting_time  = 200

        # Параметры нейросетевого анализа
        self.batch_size = 1
        self.loss_func = keras.losses.mse
        self.max_min_file = "AnomalyDetector\\modeles\\TrafficAnomalyDetector\\1.1\\M&M_traffic_VNAT.csv"
        self.feature_range = (-1, 1)

    # def NeiroAnalyze(self, characts: list):
        # print("Начинаем прогнозирование аномального трафика")
        # characts_np = np.array(characts, dtype=np.float32)
        # print(characts_np)
        # batch_x = self.normalization(characts_np[:, 6:], self.max_min_file, self.feature_range)
        # print(batch_x)
        # print(characts_np)
        #
        # exit(-1)
        # numbs_count, characts_count = batch_x.shape
        # batch_x = tf.convert_to_tensor(batch_x)
        # batch_x = tf.reshape(batch_x, (1, numbs_count, characts_count))
        #
        # dop_batch = tf.convert_to_tensor(np.zeros((4, numbs_count, characts_count), dtype=np.float32))
        # batch_x = tf.concat([batch_x, dop_batch], axis=0)
        # try:
        #     batch_x_restored = self.NetTraffic.__call__(batch_x) # verbose=0
        #     loss = np.array(self.loss_func(batch_x, batch_x_restored))[0]
        #
        #     for idx in range(numbs_count):
        #         flow_res = (int(characts_np[idx, 0]), int(characts_np[idx, 1]), int(characts_np[idx, 2]),
        #                     int(characts_np[idx, 3]), int(characts_np[idx, 4]), int(characts_np[idx, 5]),
        #                     float(loss[idx]))
        #         self.buffer.append(flow_res)
        #
        # except Exception as err:
        #     logging.exception(f"Ошибка!\n{err}")
        #     print(np.array(batch_x).shape)

    def run(self):
        with open(self.err_file, "w") as f:  # sys.stderr
            with open(self.log_file, "w") as f:  # sys.stdout

                print("Поток анализа сетевого трафика запущен!")

                # sniffer = SnifferTraffic(self.pcap_length, self.iface_name, self.path_tshark,
                #                          self.trffic_file_mask, self.path_traffic_analysis)
                # sniffer.run()

                # analizator = AnalyzerPackets(self.flow_time_limit, self.charact_file_length, self.traffic_waiting_time,
                #                              self.charact_file_name, self.ip_client, self.path_traffic_analysis)
                # analizator.run()


                traffic_pd = pd.read_csv("D:\\Пользователи\\Admin\\Рабочий стол\\"
                                         "Статья по КБ\\RATDetect\\WorkDirectory\\traffic_characters.csv")
                traffic_pd = traffic_pd[((traffic_pd["Flow_Charact.Src_IP_Flow"] != 3232270593) &
                                         (traffic_pd["Flow_Charact.Dst_IP_Flow"] != 3232270593)) &
                                        ((traffic_pd["Flow_Charact.Src_IP_Flow"] == 3232238208) |
                                         (traffic_pd["Flow_Charact.Dst_IP_Flow"] == 3232238208))]

                # Выявленные ненужные признаки:
                traffic_pd = traffic_pd.drop(["Flow_Charact.Len_Headers_Fwd"], axis=1)
                traffic_pd = traffic_pd.drop(["Flow_Charact.Std_Len_Fwd_Packets"], axis=1)
                traffic_pd = traffic_pd.drop(["Flow_Charact.Count_Flags_URG"], axis=1)
                traffic_pd = traffic_pd.drop(["Flow_Charact.Count_Flags_URG_Bwd"], axis=1)
                traffic_pd = traffic_pd.drop(["Flow_Charact.Count_Flags_URG_Fwd"], axis=1)
                traffic_pd = traffic_pd.drop(["Flow_Charact.Std_Active_Time_Flow"], axis=1)
                traffic_pd = traffic_pd.drop(["Flow_Charact.Std_InActive_Time_Flow"], axis=1)
                traffic_pd = traffic_pd.drop(["Flow_Charact.Std_Time_Diff_Fwd_Pkts"], axis=1)

                netflows = NetFlowGen(self.max_min_file, self.feature_range, self.batch_size, self.window_size)
                netflows.appendTraffic(traffic_pd)

                batch_flow_head = list()
                batch_dataflow  = None
                for batch in netflows:
                    flow_head, dataflow_tf = batch
                    batch_flow_head.append(flow_head)
                    if batch_dataflow is None:
                        batch_dataflow = dataflow_tf
                    else:
                        batch_dataflow = tf.concat([batch_dataflow, dataflow_tf], axis=0)

                    if len(batch_flow_head) == 5:
                        batch_x_restored = self.NetTraffic.__call__(batch_dataflow)  # verbose=0
                        loss = np.array(self.loss_func(batch_dataflow, batch_x_restored))

                        for i in range(len(batch_flow_head)):
                            for idx in range(self.window_size):
                                flow_res = (int(batch_flow_head[i][idx][NetFlowHead.Time_Stamp_Start]),
                                            int(batch_flow_head[i][idx][NetFlowHead.Time_Stamp_End]),
                                            int(batch_flow_head[i][idx][NetFlowHead.Src_IP_Flow]),
                                            int(batch_flow_head[i][idx][NetFlowHead.Dst_IP_Flow]),
                                            int(batch_flow_head[i][idx][NetFlowHead.Src_Port_Flow]),
                                            int(batch_flow_head[i][idx][NetFlowHead.Dst_Port_Flow]),
                                            float(loss[i][idx]))
                            self.buffer.append(flow_res)
                        batch_flow_head.clear()
                        batch_dataflow = None


                # dataset = TrainingDatasetNetFlowTrafficGen(traffic_pd, self.max_min_file, self.feature_range, 1, 10, 0.)
                # for step, batch_x in enumerate(dataset):
                #     dop_batch = tf.convert_to_tensor(np.zeros((4, 10, dataset.characts_count),
                #                                               dtype=np.float32))
                #     batch_x = tf.concat([batch_x, dop_batch], axis=0)
                #
                #     batch_x_restored = self.NetTraffic.__call__(batch_x)  # verbose=0
                #     loss = np.array(self.loss_func(batch_x, batch_x_restored))[0]
                #
                #     for idx in range(10):
                #
                #         flow_res = (int(traffic_np[idx]["Flow_Charact.Time_Stamp_Start"]),
                #                     int(traffic_np[idx]["Flow_Charact.Time_Stamp_End"]),
                #                     int(traffic_np[idx]["Flow_Charact.Src_IP_Flow"]),
                #                     int(traffic_np[idx]["Flow_Charact.Dst_IP_Flow"]),
                #                     int(traffic_np[idx]["Flow_Charact.Src_Port_Flow"]),
                #                     int(traffic_np[idx]["Flow_Charact.Dst_Port_Flow"]),
                #                     float(loss[idx]))
                #         self.buffer.append(flow_res)

                # while True:
                #     files_characts = GetFilesCSV(self.path_traffic_analysis)
                #
                #     if len(files_characts) > 0:
                #         print(f"Обнаружено {len(files_characts)} файлов с характеристиками трафика.")
                #         print("Загружаем данные о трафике")
                #
                #         characts_all = None
                #         for file in files_characts:
                #             if characts_all is None:
                #                 characts_all = pd.read_csv(file)
                #             else:
                #                 temp = pd.read_csv(file)
                #                 characts_all = pd.concat([characts_all, temp], ignore_index=True)
                #             # Path(file).unlink()
                #
                #         self.NeiroAnalyze(characts_all)
                #         break
                #
                #     else:
                #         print("Ждём данные о трафике")
                #         time.sleep(5)


if __name__ == '__main__':
    path_net_traffic = "AnomalyDetector\\modeles\\TrafficAnomalyDetector\\1.1\\model_TAD_v1.1"

    autoencoder = tf.keras.models.load_model(path_net_traffic)

    buffer = list()
    traffic_analysis = TrafficAnalyser(buffer, autoencoder, "WorkDirectory\\", "", [])
    traffic_analysis.run()

    loss = list()
    for r in buffer:
        _, _, _, _, _, _, l = r
        loss.append(l)

    mng = plt.get_current_fig_manager()
    mng.window.showMaximized()

    len_metrix = len(loss)
    plt.title(f"График аномалий в сетевом трафике")
    plt.grid(which='major')
    plt.grid(which='minor', linestyle=':')

    plt.plot([0.2 for _ in range(len_metrix)], label="Уровень нормальных данных", color="tab:red")
    plt.plot(loss, label="Обнаруженные аномалии", color="tab:blue")

    plt.legend(fontsize=10)
    plt.tight_layout()
    plt.show()


