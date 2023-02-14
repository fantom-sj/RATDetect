from NetTrafficAnalis.StreamingTrafficAnalyzer import SnifferTraffic, AnalyzerPackets, HUNDREDS_OF_NANOSECONDS
from AnomalyDetector.AutoEncoder_RNN import Autoencoder
from AuxiliaryFunctions import GetFilesCSV

from tensorflow import keras
from keras.utils import Progbar
from keras import Model

from multiprocessing import Process
from ipaddress import IPv4Address
from threading import Thread
from pathlib import Path

import matplotlib.pyplot as plt
import tensorflow as tf
import pandas as pd
import numpy as np
import logging
import time


def SnifferAnalyzerTraffic(pcap_length, iface_name, path_tshark, trffic_file_mask, path_traffic_analysis,
                           flow_time_limit, charact_file_length, traffic_waiting_time, charact_file_name, ip_client):
    # sniffer = SnifferTraffic(pcap_length, iface_name, path_tshark,
    #                          trffic_file_mask, path_traffic_analysis)
    # sniffer.start()

    analizator = AnalyzerPackets(flow_time_limit, charact_file_length, traffic_waiting_time,
                                 charact_file_name, ip_client, path_traffic_analysis)
    analizator.start()


class TrafficAnalyser(Thread):
    def __init__(self, buffer: list, NetTraffic: Model, path_traffic_analysis: str,
                 iface_name: str, ip_client: list, pcap_length=2000, window_size=10):
        super().__init__()

        self.run_toggle = False

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
        self.ip_client_number      = [int.from_bytes(IPv4Address(ip).packed, byteorder='big') for ip in ip_client]
        self.flow_time_limit       = 1 * 60 * HUNDREDS_OF_NANOSECONDS
        self.traffic_waiting_time  = 10

        # Параметры нейросетевого анализа
        self.batch_size = 1
        self.loss_func = keras.losses.mse

        max_min_file = "AnomalyDetector\\modeles\\TrafficAnomalyDetector\\1.6.2\\M&M_traffic_VNAT.csv"
        read_min_max = pd.read_csv(max_min_file)
        self.data_max = read_min_max.iloc[0].to_numpy(dtype=np.float32)
        self.data_min = read_min_max.iloc[1].to_numpy(dtype=np.float32)

        self.netflows       = dict()
        self.global_windows = list()

        self.feature_range = (-1, 1)

    def normalization(self, np_data: np.array):
        row_max, col_max = np_data.shape

        min_f, max_f = self.feature_range
        for row_idx in range(row_max):
            for col_idx in range(col_max):
                if self.data_max[col_idx] == self.data_min[col_idx]:
                    np_data[row_idx, col_idx] = min_f
                    continue
                elif np_data[row_idx, col_idx] == np.inf:
                    np_data[row_idx, col_idx] = max_f
                else:
                    np_data[row_idx, col_idx] = (np_data[row_idx, col_idx] - self.data_min[col_idx]) / \
                                                (self.data_max[col_idx] - self.data_min[col_idx])
                    np_data[row_idx, col_idx] = np_data[row_idx, col_idx] * (max_f - min_f) + min_f

        return np_data

    def split_netflow(self, np_data):
        row_max, _ = np_data.shape
        for row_idx in range(row_max):
            Src_IP_Flow   = np_data[row_idx, 2]
            Dst_IP_Flow   = np_data[row_idx, 3]
            Src_Port_Flow = np_data[row_idx, 4]
            Dst_Port_Flow = np_data[row_idx, 5]

            if Src_IP_Flow in self.ip_client_number:
                port = Dst_Port_Flow
            elif Dst_IP_Flow in self.ip_client_number:
                port = Src_Port_Flow
            else:
                continue

            flow_name = frozenset({Src_IP_Flow, Dst_IP_Flow, port})
            if not (flow_name in self.netflows):
                self.netflows[flow_name] = list()

            self.netflows[flow_name].append(np_data[row_idx])

    def portioning(self):
        for flow in self.netflows:
            batch = list()
            while len(self.netflows[flow]) > 0:
                ch = self.netflows[flow].pop(0)
                if len(batch) < self.window_size:
                    batch.append(ch)
                else:
                    self.global_windows.append(batch.copy())
                    batch.pop(0)
                    batch.append(ch)
            if len(batch) > 0:
                self.global_windows.append(batch.copy())
                batch.pop(0)

        local_windows = list()
        for i in range(len(self.global_windows)):
            if len(self.global_windows[i]) == self.window_size:
                local_windows.append(self.global_windows[i].copy())
                self.global_windows[i].clear()
            else:
                idx = 0
                while len(self.global_windows[i]) != self.window_size:
                    self.global_windows[i].append(self.global_windows[i][idx])
                    idx += 1
                local_windows.append(self.global_windows[i])

        self.global_windows = [e for e in self.global_windows if e]
        return local_windows

    def NeiroAnalyze(self, traffic_pd: pd.DataFrame):
        traffic_pd = traffic_pd[((traffic_pd["Flow_Charact.Src_IP_Flow"] != 3232270593) &
                                 (traffic_pd["Flow_Charact.Dst_IP_Flow"] != 3232270593)) &
                                ((traffic_pd["Flow_Charact.Src_IP_Flow"] == 3232238208) |
                                 (traffic_pd["Flow_Charact.Dst_IP_Flow"] == 3232238208))]

        # Выявленные ненужные признаки:
        traffic_pd = traffic_pd.drop(["Flow_Charact.Count_Flags_URG"], axis=1)
        traffic_pd = traffic_pd.drop(["Flow_Charact.Count_Flags_URG_Bwd"], axis=1)
        traffic_pd = traffic_pd.drop(["Flow_Charact.Count_Flags_URG_Fwd"], axis=1)

        traffic_np = traffic_pd.to_numpy(copy=True)
        self.split_netflow(traffic_np)
        local_windows = self.portioning()
        try:
            traffic_head    = np.array(local_windows)[:, :, :6]
            traffic_data    = np.array(local_windows)[:, :, 6:]
        except IndexError:
            print("Небыло выделено ни одного потока из трафика, ждём дальше")
            return

        progress_bar = Progbar(len(traffic_data), stateful_metrics=["Уровень аномалий"])

        for idx in range(len(traffic_data)):
            batch_x = self.normalization(traffic_data[idx])
            batch_x = tf.convert_to_tensor(batch_x, dtype=tf.float32)
            try:
                batch_x = tf.reshape(batch_x, (1, 10, 56))
                batch_x_restored = self.NetTraffic.__call__(batch_x)
                batch_x          = tf.reshape(batch_x, (10, 56))
                batch_x_restored = tf.reshape(batch_x_restored, (10, 56))
                loss             = np.array(self.loss_func(batch_x, batch_x_restored))

                values = [("Уровень аномалий", loss)]
                progress_bar.add(1, values=values)

                i = np.argmax(loss)
                flow_res = (int(traffic_head[idx, i, 0]),
                            int(traffic_head[idx, i, 1]),
                            int(traffic_head[idx, i, 2]),
                            int(traffic_head[idx, i, 3]),
                            int(traffic_head[idx, i, 4]),
                            int(traffic_head[idx, i, 5]),
                            loss[i])
                self.buffer.append(flow_res)
            except Exception as err:
                logging.exception(f"\nОшибка!\n{batch_x}\n{err}\n")
                continue

    def run(self):
        for file in Path(self.path_traffic_analysis).iterdir():
            if ".pcap" in str(file):
                file.unlink()
            elif "tmp" in str(file):
                for tmp_file in Path(self.path_traffic_analysis + "\\tmp").iterdir():
                    tmp_file.unlink()
            elif ".csv" in str(file):
                path_new = self.path_traffic_analysis + "Обработанные файлы"
                if not Path(path_new).exists():
                    Path(path_new).mkdir()
                file_only_name = str(file).split("\\")[-1]
                Path(file).rename(path_new + "\\" + file_only_name)

        with open(self.err_file, "w") as f:  # sys.stderr
            with open(self.log_file, "w") as f:  # sys.stdout

                print("Поток анализа сетевого трафика запущен!")

                snif_analiz_proc = Process(target=SnifferAnalyzerTraffic,
                                           args=(self.pcap_length, self.iface_name, self.path_tshark,
                                                 self.trffic_file_mask, self.path_traffic_analysis,
                                                 self.flow_time_limit, self.charact_file_length,
                                                 self.traffic_waiting_time, self.charact_file_name, self.ip_client,))
                snif_analiz_proc.start()

                # traffic_pd = pd.read_csv("D:\\Пользователи\\Admin\\Рабочий стол\\"
                #                          "Статья по КБ\\RATDetect\\WorkDirectory\\traffic_characters.csv")
                # self.NeiroAnalyze(traffic_pd)

                while self.run_toggle:
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

                            path_new = self.path_traffic_analysis + "Обработанные файлы"
                            if not Path(path_new).exists():
                                Path(path_new).mkdir()
                            file_only_name = file.split("\\")[-1]
                            Path(file).rename(path_new + "\\" + file_only_name)

                        self.NeiroAnalyze(characts_all)
                        characts_all = characts_all.iloc[0:0]
                        # break
                    else:
                        time.sleep(5)

                print("Поток анализа сетевого трафика остановлен!")
                snif_analiz_proc.kill()


def convert_batch_model():
    path_net_traffic = "AnomalyDetector\\modeles\\TrafficAnomalyDetector\\1.6.2\\Checkpoint\\epoch_2"
    window_size = 10

    encoder = {"1_Input": (window_size, 51), "2_GRU_seq": (43, 51), "3_GRU_seq": (35, 43),
               "4_GRU_seq": (27, 35), "5_GRU_seq": (19, 27), "6_GRU_seq": (11, 19)}
    decoder = {"7_Input": (window_size, 11), "8_GRU_seq": (19, 11), "9_GRU_seq": (27, 19),
               "10_GRU_seq": (35, 27), "11_GRU_seq": (43, 35), "12_GRU_seq": (51, 43)}
    arhiteche = (encoder, decoder)

    autoencoder = Autoencoder(51, arhiteche, window_size, 1)
    autoencoder.build((1, window_size, 51))
    autoencoder.summary()
    autoencoder.encoder_model.summary()
    autoencoder.decoder_model.summary()
    optimizer = keras.optimizers.Adam(learning_rate=0.0001)
    autoencoder.compile(optimizer=optimizer, loss=keras.losses.mse)

    autoencoder.load_weights(path_net_traffic)
    autoencoder.__call__(tf.convert_to_tensor(np.random.random(51*window_size).reshape((1, window_size, 51)),
                                              dtype=tf.float32))
    autoencoder.save("AnomalyDetector\\modeles\\TrafficAnomalyDetector\\1.6.1\\model_TAD_v1.6.1_one")


if __name__ == '__main__':
    tf.config.set_visible_devices([], 'GPU')

    path_net_traffic = "AnomalyDetector\\modeles\\TrafficAnomalyDetector\\1.6.2\\Checkpoint\\epoch_2"
    autoencoder = tf.keras.models.load_model(path_net_traffic)

    buffer = list()
    traffic_analysis = TrafficAnalyser(buffer, autoencoder, "WorkDirectory\\",
                                       "VMware_Network_Adapter_VMnet3", [IPv4Address("192.168.10.128")], window_size=10)
    traffic_analysis.run()

    porog = 0.0

    loss = list()
    for r in buffer:
        _, _, src_ip, dst_ip, src_port, dst_port, l = r
        if l > porog:
            print(f"IP источника: {IPv4Address(src_ip)}\tIP назначения: {IPv4Address(dst_ip)}\t"
                  f"Порт источника: {src_port}\t Порт назначения: {dst_port}\tАномальность: {l}")
        loss.append(l)

    mng = plt.get_current_fig_manager()
    mng.window.showMaximized()

    len_metrix = len(loss)
    plt.title(f"График аномалий в сетевом трафике")
    plt.grid(which='major')
    plt.grid(which='minor', linestyle=':')
    plt.ylim([-0.01, 1.0])

    plt.plot([porog for _ in range(len_metrix)], label="Уровень нормальных данных", color="tab:red")
    plt.plot(loss, label="Обнаруженные аномалии", color="tab:blue")

    plt.legend(fontsize=10)
    plt.tight_layout()
    plt.show()


