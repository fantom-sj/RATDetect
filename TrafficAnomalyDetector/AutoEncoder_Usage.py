import tensorflow as tf
from scipy.signal import savgol_filter
from tensorflow import keras
from keras.utils import Progbar
from ipaddress import IPv4Address

from AutoEncoder_RNN import TrainingDatasetGen
from SnifferPaket.StreamingTrafficAnalyzer import Analyzer
from SnifferPaket.TestDatasetCreate import TrafficTestGen, sequence, increasingly, random, descending

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import logging


def CreateTestDataset(window_size):
    path_name = "F:\\VNAT\\Mytraffic\\youtube_me"
    trffic_name = "test_dataset"
    normal_name = "test_dataset_narmal"
    anomal_name = "test_dataset_anomaly"

    charact_file_length = 10000000
    charact_file_name = "test_dataset_2"
    ip_client = [IPv4Address("192.168.10.128")]

    paradigma = [(10000, 0, sequence),
                 (300, 0.06, increasingly), (5000, 3000, random), (300, 0.06, descending),
                 (10000, 0, sequence), (0, 5000, sequence),
                 (10000, 0, sequence), (10000, 10000, random),
                 (10000, 0, sequence), (100, 0.3, increasingly), (5000, 1000, random), (200, 0.02, descending),
                 (10000, 0, sequence), (200, 0.1, increasingly), (200, 0.1, descending),
                 (10000, 0, sequence), (200, 0.02, increasingly), (200, 0.02, descending), (10000, 0, sequence)]

    # paradigma = [(2000, 0, sequence), (300, 0.06, increasingly), (300, 0.06, descending), (2000, 0, sequence)]

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

    analizator = Analyzer(window_size, charact_file_length, charact_file_name, ip_client, path_name, trffic_name)
    analizator.PaketsAnalyz(pakets)

    generator.PrintGarafAnomaly()

    return generator.counts_anomal


def main():

    # Парамеры автоэнкодера
    versia          = "0.8.6.2"
    batch_size      = 100
    window_size     = 1000
    loss_func       = keras.losses.mse
    max_min_file    = "modeles\\TrafficAnomalyDetector\\" + versia + "\\M&M_traffic_VNAT.csv"
    model           = "modeles\\TrafficAnomalyDetector\\" + versia + "\\model_TAD_v" + versia

    # Тестовый датасет
    real_anomaly = CreateTestDataset(window_size)

    characts_file   = "F:\\VNAT\\Mytraffic\\youtube_me\\test_dataset\\test_dataset_21.csv"
    feature_range   = (-1, 1)
    caracts_data    = pd.read_csv(characts_file)
    caracts_pd      = caracts_data.drop(["Time_Stamp"], axis=1)
    caracts_pd      = caracts_pd.drop(["Count_src_is_dst_ports"], axis=1)
    caracts_pd      = caracts_pd.drop(["Dev_size_TCP_paket"], axis=1)
    caracts_pd      = caracts_pd.drop(["Dev_size_UDP_paket"], axis=1)
    caracts_pd      = caracts_pd.drop(["Dev_client_paket_size"], axis=1)
    caracts_pd      = caracts_pd.drop(["Dev_server_paket_size"], axis=1)
    caracts_numpy   = TrainingDatasetGen.normalization(caracts_pd, max_min_file, feature_range).to_numpy()

    numbs_count, caracts_count    = caracts_numpy.shape
    batch_count                   = round(numbs_count/batch_size) - 11

    # Определение автоэнкодера
    autoencoder = tf.keras.models.load_model(model)
    print("Модель загружена")

    print("Начинаем прогнозирование аномального трафика.")
    metrics_analiz = {}

    valid_metrics_name = ["Расхождение"]
    progress_bar = Progbar(batch_count, stateful_metrics=valid_metrics_name)

    for idx in range(0, batch_count, 1):
        batch_x = []
        for i in range(batch_size):
            batch_x.append(caracts_numpy[i + (idx * batch_size):window_size + i + (idx * batch_size)])
        try:
            batch_x = tf.convert_to_tensor(batch_x)
            # batch_x = tf.reshape(batch_x, (1, windows_size, caracts_count))
            batch_x_restored = autoencoder.predict(batch_x, verbose=0)

            loss = loss_func(batch_x, batch_x_restored)
            loss = tf.math.reduce_mean(loss, 1)
            if idx == 0:
                metrics_analiz["loss"] = loss
            else:
                metrics_analiz["loss"] = tf.concat([metrics_analiz["loss"], loss], axis=0)
            mean_loss = tf.math.multiply(tf.math.reduce_mean(loss), tf.constant(100, dtype=tf.float32))
            values = [("Расхождение", mean_loss)]
            progress_bar.add(1, values=values)
        except Exception as err:
            logging.exception(f"Ошибка!\n{err}")
            print(np.array(batch_x).shape)
            continue

        metrics_analiz["loss"] = \
            savgol_filter(np.array(metrics_analiz["loss"]), 31, 3)

    metrics_analiz_pd = TrainingDatasetGen.normalization(pd.DataFrame(metrics_analiz), feature_range=(0, 100))
    metrics_analiz_norm = metrics_analiz_pd.to_dict("list")

    mng = plt.get_current_fig_manager()
    mng.window.showMaximized()

    len_metrix = len(metrics_analiz_norm["loss"])
    plt.xlim([-5.0, len_metrix + 5])
    plt.ylim([-5.0, 105.0])
    plt.title(f"График аномалий в сетевом трафике")   #, версия: {versia}
    plt.grid(which='major')
    plt.grid(which='minor', linestyle=':')

    plt.plot(real_anomaly, label="Реальные аномалии", color="tab:red")
    plt.plot(metrics_analiz_norm["loss"], label="Обнаруженные аномалии", color="tab:blue")

    plt.legend(fontsize=10)
    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    main()