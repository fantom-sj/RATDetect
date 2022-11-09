from scipy.signal import savgol_filter
# from StreamingTrafficAnalyzer import Analyzer
from ipaddress import IPv4Address
from SnifferPaket.characts import ParseTraffic
from pathlib import Path

import matplotlib.pyplot as plt
import random as rnd
import pandas as pd
import logging

sequence     = 0
increasingly = 1
random       = 2
descending   = 3
mixing       = 4
linear_up    = 5
linear_down  = 6


class TrafficTestGen:
    def __init__(self, window_size, trffic_normal_name, trffic_anomal_name, path_name):
        self.window_size        = window_size
        self.trffic_normal_name = trffic_normal_name
        self.trffic_anomal_name = trffic_anomal_name
        self.path_name          = path_name

        self.normal_pakets  = []
        self.anomaly_pakets = []
        self.array_pakets   = []
        self.pakets         = None

    def SetBasicTraffic(self, normal_pcap_files, anomaly_pcap_files):
        for file in normal_pcap_files:
            file_traffic = self.path_name + "\\" + self.trffic_normal_name + "\\" + file
            if not Path(file_traffic).exists():
                logging.exception(f"Ошибка: файла траффика с именем {file} не существует!")
                continue

            for pkt in ParseTraffic(file_traffic):
                self.normal_pakets.append(pkt)
        print(f"Всего собрано {len(self.normal_pakets)} пакетов с нормальным трафиком")

        for file in anomaly_pcap_files:
            file_traffic = self.path_name + "\\" + self.trffic_anomal_name + "\\" + file
            if not Path(file_traffic).exists():
                logging.exception(f"Ошибка: файла траффика с именем {file_traffic} не существует!")
                continue

            for pkt in ParseTraffic(file_traffic):
                self.anomaly_pakets.append(pkt)
        print(f"Всего собрано {len(self.anomaly_pakets)} пакетов с аномальным трафиком")

    def MixPakets(self, paradigmas):
        idx_normal = 0
        idx_anomal = 0

        for paradigma in paradigmas:
            x, y, z = paradigma

            if z == sequence:
                for i in range(x):
                    pkt = self.normal_pakets[idx_normal]
                    pkt["metka"] = "Normal"
                    self.array_pakets.append(pkt)
                    idx_normal += 1
                for i in range(y):
                    pkt = self.anomaly_pakets[idx_anomal]
                    pkt["metka"] = "Anomal"
                    self.array_pakets.append(pkt)
                    idx_anomal += 1

            elif z == increasingly:
                num_add = 1.0
                interval = round(1/y)
                for i in range(x):
                    for j in range(interval):
                        pkt = self.normal_pakets[idx_normal]
                        pkt["metka"] = "Normal"
                        self.array_pakets.append(pkt)
                        idx_normal += 1
                    for j in range(round(num_add)):
                        pkt = self.anomaly_pakets[idx_anomal]
                        pkt["metka"] = "Anomal"
                        self.array_pakets.append(pkt)
                        idx_anomal += 1
                    num_add += y

            elif z == descending:
                num_sub = 1 + y * x
                interval = round(1 / y)
                for i in range(x):
                    for j in range(interval):
                        pkt = self.normal_pakets[idx_normal]
                        pkt["metka"] = "Normal"
                        self.array_pakets.append(pkt)
                        idx_normal += 1
                    for j in range(round(num_sub)):
                        pkt = self.anomaly_pakets[idx_anomal]
                        pkt["metka"] = "Anomal"
                        self.array_pakets.append(pkt)
                        idx_anomal += 1
                    num_sub -= y

            elif z == random:
                add_normal = 0
                add_anomal = 0
                while add_normal != x and add_anomal != y:
                    coin = rnd.randint(0, 1)
                    if coin == 0 and add_normal != x:
                        pkt = self.normal_pakets[idx_normal]
                        pkt["metka"] = "Normal"
                        self.array_pakets.append(pkt)
                        idx_normal += 1
                        add_normal += 1
                    elif coin == 1 and add_anomal != y:
                        pkt = self.anomaly_pakets[idx_anomal]
                        pkt["metka"] = "Anomal"
                        self.array_pakets.append(pkt)
                        idx_anomal += 1
                        add_anomal += 1
                    else:
                        continue

            elif z == mixing:
                if x > y:
                    interval = round(x/y)
                    passed = 0
                    for i in range(x):
                        pkt = self.normal_pakets[idx_normal]
                        pkt["metka"] = "Normal"
                        self.array_pakets.append(pkt)
                        idx_normal += 1
                        passed += 1
                        if passed == interval:
                            pkt = self.anomaly_pakets[idx_anomal]
                            pkt["metka"] = "Anomal"
                            self.array_pakets.append(pkt)
                            idx_anomal += 1
                            passed = 0
                elif y > x:
                    interval = round(y/x)
                    passed = 0
                    for i in range(y):
                        pkt = self.anomaly_pakets[idx_anomal]
                        pkt["metka"] = "Anomal"
                        self.array_pakets.append(pkt)
                        idx_anomal += 1
                        if passed == interval:
                            pkt = self.normal_pakets[idx_normal]
                            pkt["metka"] = "Normal"
                            self.array_pakets.append(pkt)
                            idx_normal += 1
                            passed += 1
                elif x == y:
                    for i in range(x):
                        pkt = self.normal_pakets[idx_normal]
                        pkt["metka"] = "Normal"
                        self.array_pakets.append(pkt)
                        idx_normal += 1

                        pkt = self.anomaly_pakets[idx_anomal]
                        pkt["metka"] = "Anomal"
                        self.array_pakets.append(pkt)
                        idx_anomal += 1
                else:
                    continue

            elif z == linear_up:
                for i in range(x):
                    for i in range(y):
                        pkt = self.normal_pakets[idx_normal]
                        pkt["metka"] = "Normal"
                        self.array_pakets.append(pkt)
                        idx_normal += 1

                    for j in range(y):
                        pkt = self.anomaly_pakets[idx_anomal]
                        pkt["metka"] = "Anomal"
                        self.array_pakets.append(pkt)
                        idx_anomal += 1

                    y += 1

            else:
                continue

        self.pakets = pd.DataFrame(self.array_pakets)
        return self.array_pakets

    def PrintGarafAnomaly(self):
        anomaly = self.pakets["metka"].to_numpy()
        len_anomaly, = anomaly.shape
        new_len_anomaly = round(len_anomaly / self.window_size) * self.window_size - self.window_size
        anomaly_ = anomaly[:new_len_anomaly]

        self.counts_anomal = []
        for i in range(self.window_size, new_len_anomaly, 1):
            count_anomal = 0
            for j in range(i - self.window_size, i, 1):
                if anomaly_[j] == "Anomal":
                    count_anomal += 1
            self.counts_anomal.append(count_anomal/10)

        # counts_anomal_smooth = savgol_filter(counts_anomal, 31, 3)

        plt.xlim([-10.0, new_len_anomaly + 10.0])
        plt.ylim([-5.0, 105.0])
        plt.title(f"График аномалий в созданном трафике")
        plt.grid(which='major')
        plt.grid(which='minor', linestyle=':')

        plt.plot(self.counts_anomal, label="Найденные аномалии", color="tab:red")
        # plt.plot(counts_anomal_smooth, label="Сглаженные аномалии", color="tab:blue")

        plt.legend(fontsize=10)
        plt.tight_layout()
        plt.show()


if __name__ == '__main__':
    path_name   = "F:\\VNAT\\Mytraffic\\youtube_me"
    trffic_name = "test_dataset"
    normal_name = "test_dataset_narmal"
    anomal_name = "test_dataset_anomaly"

    window_size = 1000
    charact_file_length = 10000000
    charact_file_name = "test_dataset_1"
    ip_client = [IPv4Address("192.168.10.128")]

    paradigma = [(10000, 0, sequence),
                 (300, 0.06, increasingly), (5000, 3000, random), (300, 0.06, descending),
                 (10000, 0, sequence), (0, 5000, sequence),
                 (10000, 0, sequence), (10000, 10000, random),
                 (10000, 0, sequence), (100, 0.3, increasingly), (5000, 1000, random), (200, 0.02, descending),
                 (10000, 0, sequence), (200, 0.1, increasingly), (200, 0.1, descending),
                 (10000, 0, sequence), (200, 0.02, increasingly), (200, 0.02, descending), (10000, 0, sequence)]

    normal_pcap_files = []
    for i in range(511, 611, 1):
        normal_pcap_files.append("traffic_" + str(i) + ".pcapng")

    anomal_pcap_files = []
    for i in range(0, 55, 1):
        anomal_pcap_files.append("traffic_" + str(i) + ".pcapng")

    generator = TrafficTestGen(window_size, normal_name, anomal_name, path_name)
    generator.SetBasicTraffic(normal_pcap_files, anomal_pcap_files)
    pakets = generator.MixPakets(paradigma)
    print(f"Получен массив из {len(pakets)} сетевых пакетов")

    # analizator = Analyzer(window_size, charact_file_length, charact_file_name, ip_client, path_name, trffic_name)
    # analizator.PaketsAnalyz(pakets)

    generator.PrintGarafAnomaly()


