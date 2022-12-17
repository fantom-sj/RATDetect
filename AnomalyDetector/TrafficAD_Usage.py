import tensorflow as tf
from scipy.signal import savgol_filter
from tensorflow import keras
from keras.utils import Progbar
from ipaddress import IPv4Address

from AutoEncoder import loss_for_vae, noiser
from AutoEncoder_RNN import TrainingDatasetGen
from NetTrafficAnalis.TrafficСharacts import HUNDREDS_OF_NANOSECONDS
from NetTrafficAnalis.StreamingTrafficAnalyzer import AnalyzerPackets
from NetTrafficAnalis.TestDatasetCreate import TrafficTestGen, sequence, increasingly, random, descending

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import logging


def CreateTestDataset(window_size):
    path_name = "F:\\VNAT\\Mytraffic\\youtube_me"
    normal_name = "test_dataset_narmal"
    anomal_name = "test_dataset_anomaly"

    flow_time_limit = 1 * 60 * HUNDREDS_OF_NANOSECONDS
    traffic_waiting_time = 200
    charact_file_length = 10000000
    charact_file_mask = "test_dataset_3"
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

    analizator = AnalyzerPackets(flow_time_limit, charact_file_length, traffic_waiting_time, charact_file_mask, ip_client, path_name)
    analizator.PaketsAnalyz(pakets)

    generator.PrintGarafAnomaly()

    return generator.counts_anomal


def main():

    # Парамеры автоэнкодера
    versia          = "0.9.3_vae"
    batch_size      = 1
    window_size     = 1
    # loss_func       = AutoencoderBase.loss_for_vae
    max_min_file    = "modeles\\TrafficAnomalyDetector\\" + versia + "\\M&M_traffic_VNAT.csv"
    model           = "modeles\\TrafficAnomalyDetector\\" + versia + "\\model_TAD_v" + versia
    hidden_space    = 15

    # Тестовый датасет
    # real_anomaly = CreateTestDataset(window_size)

    flow_anomal_res = {}
    path_name   = "D:\\Пользователи\\Admin\\Рабочий стол\\Статья по КБ\\RATDetect\\WorkDirectory"

    flow_time_limit = 1 * 60 * HUNDREDS_OF_NANOSECONDS
    traffic_waiting_time = 200
    charact_file_length = 10000000
    charact_file_mask = "test_dataset_"
    ip_client = [IPv4Address("192.168.10.128")]

    analizator = AnalyzerPackets(flow_time_limit, charact_file_length, traffic_waiting_time, charact_file_mask,
                                 ip_client, path_name)
    analizator.GetFilesTraffic()
    analizator.ProcessingTraffic(analizator.files_traffic_arr)
    print("Анализ трафика завершен")
    charact_index = analizator.index_charact_file

    feature_range   = (-1, 1)
    characts_data   = pd.read_csv(f"{path_name}\\{charact_file_mask}{charact_index}.csv")
    characts_np     = characts_data.to_dict("records")
    for flow in characts_np:
        name_flow = f'{IPv4Address(flow["Flow_Charact.Src_IP_Flow"])}-' \
                    f'{IPv4Address(flow["Flow_Charact.Dst_IP_Flow"])}-' \
                    f'{flow["Flow_Charact.Src_Port_Flow"]}-' \
                    f'{flow["Flow_Charact.Dst_Port_Flow"]}'
        flow_anomal_res[name_flow] = None

    characts_pd     = characts_data.drop(["Flow_Charact.Time_Stamp_Start"], axis=1)
    characts_pd     = characts_pd.drop(["Flow_Charact.Time_Stamp_End"], axis=1)
    characts_pd     = characts_pd.drop(["Flow_Charact.Src_IP_Flow"], axis=1)
    characts_pd     = characts_pd.drop(["Flow_Charact.Dst_IP_Flow"], axis=1)
    characts_pd     = characts_pd.drop(["Flow_Charact.Src_Port_Flow"], axis=1)
    characts_pd     = characts_pd.drop(["Flow_Charact.Dst_Port_Flow"], axis=1)
    characts_numpy  = TrainingDatasetGen.normalization(characts_pd, max_min_file, feature_range, True)

    numbs_count, characts_count    = characts_numpy.shape
    batch_count                   = round(numbs_count/batch_size)

    # Определение автоэнкодера
    autoencoder = tf.keras.models.load_model(model, custom_objects={"noiser": noiser})
    print("Модель загружена")

    print("Начинаем прогнозирование аномального трафика.")
    metrics_analiz = {}

    valid_metrics_name = ["Расхождение"]
    progress_bar = Progbar(batch_count, stateful_metrics=valid_metrics_name)

    for idx in range(0, batch_count, 1):
        batch_x = []
        for i in range(batch_size):
            batch_x.append(characts_numpy[i + (idx * batch_size):window_size + i + (idx * batch_size)])
            loss = 0
        try:
            batch_x = tf.convert_to_tensor(batch_x)
            # batch_x = tf.reshape(batch_x, (1, windows_size, characts_count))
            logits = autoencoder.predict(batch_x, verbose=0)
            z_mean_res = tf.gather(logits, [i for i in range(0, hidden_space)], axis=-1)
            z_log_var_res = tf.gather(logits, [i for i in range(hidden_space, hidden_space * 2)], axis=-1)
            batch_x_restored = tf.gather(logits, [i for i in range(hidden_space * 2,
                                                  characts_count + (hidden_space * 2))], axis=-1)
            loss = loss_for_vae(batch_x, batch_x_restored, (z_mean_res, z_log_var_res,
                                                            batch_size, characts_count))

            # loss = tf.math.reduce_mean(loss, 1)
            if idx == 0:
                metrics_analiz["loss"] = loss
            else:
                metrics_analiz["loss"] = tf.concat([metrics_analiz["loss"], loss], axis=0)
            # mean_loss = tf.math.multiply(tf.math.reduce_mean(loss), tf.constant(100, dtype=tf.float32))
            values = [("Расхождение", loss)]
            progress_bar.add(1, values=values)

        except Exception as err:
            logging.exception(f"Ошибка!\n{err}")
            print(batch_x.shape)
            print(np.array(loss))
            continue

    # metrics_analiz_norm = TrainingDatasetGen.normalization(pd.DataFrame(metrics_analiz),
    #                                                      feature_range=(0, 100), mix_max_from_file=False)

    idx = 0
    for flow_name in flow_anomal_res:
        flow_anomal_res[flow_name] = metrics_analiz["loss"][idx]
        if metrics_analiz["loss"][idx] >= 0.04:
            print(f"{flow_name}: {flow_anomal_res[flow_name]}")
        idx += 1

    mng = plt.get_current_fig_manager()
    mng.window.showMaximized()

    len_metrix = len(metrics_analiz["loss"])
    plt.xlim([-5.0, len_metrix + 5])
    plt.ylim([-5.0, 105.0])
    plt.title(f"График аномалий в сетевом трафике")
    plt.grid(which='major')
    plt.grid(which='minor', linestyle=':')

    # plt.plot(real_anomaly, label="Реальные аномалии", color="tab:red")
    plt.plot(metrics_analiz["loss"], label="Обнаруженные аномалии", color="tab:blue")

    plt.legend(fontsize=10)
    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    main()