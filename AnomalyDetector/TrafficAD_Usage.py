import tensorflow as tf
tf.config.set_visible_devices([], 'GPU')
from scipy.signal import savgol_filter
from tensorflow import keras
from keras.utils import Progbar
from ipaddress import IPv4Address

from AutoEncoder_RNN import TrainingDatasetNetFlowTrafficGen
from NetTrafficAnalis.TrafficСharacts import HUNDREDS_OF_NANOSECONDS
from NetTrafficAnalis.StreamingTrafficAnalyzer import AnalyzerPackets
from NetTrafficAnalis.TestDatasetCreate import TrafficTestGen, sequence, increasingly, random, descending
from sklearn.ensemble import IsolationForest

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import logging
import pickle
from tqdm import tqdm


def CreateTestDataset(window_size):
    path_name = "D:\\Пользователи\\Admin\\Рабочий стол\\"\
                "Статья по КБ\\RATDetect\\WorkDirectory\\traffic_characters.csv"
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
    versia          = "1.1"
    batch_size      = 5
    window_size     = 10
    loss_func       = keras.losses.mse
    path_model      = "modeles\\TrafficAnomalyDetector\\" + versia + "\\"
    max_min_file    = path_model + "M&M_traffic_VNAT.csv"
    model           = path_model + "model_TAD_v" + versia  # "Checkpoint\\epoch_1"
    anomaly         = path_model + "anomalyDetector"
    porog_anomaly   = 0.4

    # Тестовый датасет
    # real_anomaly = CreateTestDataset(window_size)

    flow_anomal_res = {}
    path_name   = "D:\\Пользователи\\Admin\\Рабочий стол\\Статья по КБ\\RATDetect\\WorkDirectory"

    flow_time_limit = 1 * 60 * HUNDREDS_OF_NANOSECONDS
    traffic_waiting_time = 200
    charact_file_length = 10000000
    charact_file_mask = "dataset_"
    ip_client = [IPv4Address("192.168.10.128")]

    # analizator = AnalyzerPackets(flow_time_limit, charact_file_length, traffic_waiting_time, charact_file_mask,
    #                              ip_client, path_name)
    # analizator.GetFilesTraffic()
    # analizator.ProcessingTraffic(analizator.files_traffic_arr)
    # print("Анализ трафика завершен")
    # charact_index = analizator.index_charact_file

    feature_range   = (-1, 1)
    characts_data   = pd.read_csv("D:\\Пользователи\\Admin\\Рабочий стол\\"
                                         "Статья по КБ\\RATDetect\\WorkDirectory\\traffic_characters.csv")
    # for charact_index in range(1):
    #     characts_data = pd.concat([characts_data, pd.read_csv(f"{path_name}\\{charact_file_mask}{charact_index}.csv")],
    #                               ignore_index=True)

    characts_data = characts_data[((characts_data["Flow_Charact.Src_IP_Flow"] != 3232270593) &
                                   (characts_data["Flow_Charact.Dst_IP_Flow"] != 3232270593)) &
                                  ((characts_data["Flow_Charact.Src_IP_Flow"] == 3232238208) |
                                   (characts_data["Flow_Charact.Dst_IP_Flow"] == 3232238208))]
    characts_data.sort_values(by="Flow_Charact.Time_Stamp_End")

    characts_np = characts_data.to_dict("records")
    characts_pd = characts_data.drop(["Flow_Charact.Time_Stamp_Start"], axis=1)
    characts_pd = characts_pd.drop(["Flow_Charact.Time_Stamp_End"], axis=1)
    # characts_pd = characts_pd.drop(["Flow_Charact.Src_IP_Flow"], axis=1)
    # characts_pd = characts_pd.drop(["Flow_Charact.Dst_IP_Flow"], axis=1)
    # characts_pd = characts_pd.drop(["Flow_Charact.Src_Port_Flow"], axis=1)
    # characts_pd = characts_pd.drop(["Flow_Charact.Dst_Port_Flow"], axis=1)

    # Выявленные ненужные признаки:
    characts_pd = characts_pd.drop(["Flow_Charact.Len_Headers_Fwd"], axis=1)
    characts_pd = characts_pd.drop(["Flow_Charact.Std_Len_Fwd_Packets"], axis=1)
    characts_pd = characts_pd.drop(["Flow_Charact.Count_Flags_URG"], axis=1)
    characts_pd = characts_pd.drop(["Flow_Charact.Count_Flags_URG_Bwd"], axis=1)
    characts_pd = characts_pd.drop(["Flow_Charact.Count_Flags_URG_Fwd"], axis=1)
    characts_pd = characts_pd.drop(["Flow_Charact.Std_Active_Time_Flow"], axis=1)
    characts_pd = characts_pd.drop(["Flow_Charact.Std_InActive_Time_Flow"], axis=1)
    characts_pd = characts_pd.drop(["Flow_Charact.Std_Time_Diff_Fwd_Pkts"], axis=1)

    dataset = TrainingDatasetNetFlowTrafficGen(characts_pd, max_min_file, feature_range,
                                          batch_size, window_size, 0.0)

    numbs_count, characts_count    = (dataset.numbs_count, dataset.characts_count)

    # Определение автоэнкодера
    autoencoder = tf.keras.models.load_model(model)
    print("Модель загружена")

    print("Начинаем прогнозирование аномального трафика.")
    metrics_analiz = {}

    valid_metrics_name = ["Расхождение"]
    progress_bar = Progbar(len(dataset), stateful_metrics=valid_metrics_name)

    for step, batch_x in enumerate(dataset):
        logits = autoencoder.__call__(batch_x)
        loss_value = loss_func(batch_x, logits)

        loss = float(np.mean(np.array(loss_value)[0]))

        # loss_tf = tf.math.reduce_mean(loss, 1)
        if step == 0:
            metrics_analiz["loss"] = loss_value
        else:
            metrics_analiz["loss"] = tf.concat([metrics_analiz["loss"], loss_value], axis=0)

        values = [("Ошибка", loss)]
        progress_bar.add(1, values=values)

    print(metrics_analiz["loss"])

    # flow_anomal_res = {}
    # real_anomaly = []
    # data = np.array(metrics_analiz["loss"])
    #
    # for idx in range(len(characts_np)):
    #     flow_name = f'{IPv4Address(characts_np[idx]["Flow_Charact.Src_IP_Flow"])}-' \
    #                 f'{IPv4Address(characts_np[idx]["Flow_Charact.Dst_IP_Flow"])}-'
    #
    #     if "129" in flow_name:
    #         real_anomaly.append(1)
    #     else:
    #         real_anomaly.append(0)
    #
    #     if not flow_name in flow_anomal_res:
    #         flow_anomal_res[flow_name] = list()
    #     flow_anomal_res[flow_name].append(data[idx])
    #
    # anomaly_level_flow = {}
    # for flow in flow_anomal_res:
    #     anomaly_level_flow[flow] = 0
    #     for loss in flow_anomal_res[flow]:
    #         if loss >= porog_anomaly:
    #             anomaly_level_flow[flow] += 1
    #
    # no_anomaly_level_flow = {}
    # for flow in flow_anomal_res:
    #     no_anomaly_level_flow[flow] = 0
    #     for loss in flow_anomal_res[flow]:
    #         if loss < porog_anomaly:
    #             no_anomaly_level_flow[flow] += 1
    #
    # print("\nУровень аномальной активности для каждого сетевого потока:")
    # for flow in anomaly_level_flow:
    #     print(f"{flow}: {anomaly_level_flow[flow]}")
    #
    # print("\nУровень нормальной активности для каждого сетевого потока:")
    # for flow in no_anomaly_level_flow:
    #     print(f"{flow}: {no_anomaly_level_flow[flow]}")
    #
    mng = plt.get_current_fig_manager()
    mng.window.showMaximized()

    len_metrix = len(metrics_analiz["loss"])
    plt.xlim([-5.0, len_metrix + 5])
    plt.ylim([-0.02, 1.01])
    plt.title(f"График аномалий в сетевом трафике")
    plt.grid(which='major')
    plt.grid(which='minor', linestyle=':')

    # plt.plot(real_anomaly, label="Обнаруженные аномалии", color="tab:red")
    plt.plot([porog_anomaly for _ in range(len_metrix)], label="Уровень нормальных данных", color="tab:red")
    plt.plot(metrics_analiz["loss"], label="Обнаруженные аномалии", color="tab:blue")

    plt.legend(fontsize=10)
    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    main()