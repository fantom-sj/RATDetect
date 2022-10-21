from tensorflow import keras
from keras.utils import Progbar
# from SnifferPaket.TrafficParcNDPI import PreprocessingPcapng
from pathlib import Path
# from SnifferPaket.characts import CHARACTERISTIC
import pylab
import time
import pandas as pd
import numpy as np


pcapng_files = [
    "traffic_RAT_NingaliNET_1.pcapng",
    "traffic_RAT_NingaliNET_2.pcapng",
    "traffic_RAT_NingaliNET_3.pcapng",
    "traffic_RAT_NingaliNET_4.pcapng",
    "traffic_RAT_NingaliNET_5.pcapng"
]


# def Preproccessing(characts_file, windows_size, path_name):
#
#     path_characts_file = Path(characts_file)
#     if path_characts_file.exists() is False:
#         pd_ch_name = pd.DataFrame()
#         for ch in CHARACTERISTIC:
#             pd_ch_name[ch] = []
#         pd_ch_name.to_csv(str(path_characts_file), index=False)
#
#
#     for file in pcapng_files:
#         res = PreprocessingPcapng(path_name,
#                                   windows_size,
#                                   ["192.168.10.128"],
#                                   path_name + file,
#                                   characts_file)
#         if res:
#             print("Парсинг выполнен успешно!\n")
#         else:
#             print("Парсинг не выполнен!\n")
#     return 0


def normalization(pd_data, max_min_pd):
    max_min_pd = max_min_pd.to_numpy()
    data_max = max_min_pd[0]
    data_min = max_min_pd[1]

    index = 0
    for col in pd_data:
        if col != "Time_Stamp":
            pd_data[col] = (pd_data[col] - data_min[index]) / (data_max[index] - data_min[index])
            index += 1
    return pd_data


def main():
    # time.sleep(7500)
    versia = "0.8.2"
    max_min_file = "modeles\\TrafficAnomalyDetector\\" + versia + "\\M&M_traffic_VNAT.csv"
    characts_file = "..\\data\\pcap\\test_dataset\\array_characts.csv"
    model_name = "modeles\\TrafficAnomalyDetector\\" + versia + "\\model_TAD_v" + versia + "_e1"

    windows_size = 1000
    caracts_pd = pd.read_csv(characts_file)
    caracts_pd = caracts_pd.drop(["Time_Stamp"], axis=1)
    caracts_pd = caracts_pd.drop(["RAT_count"], axis=1)

    min_and_max_pd = pd.read_csv(max_min_file)
    caracts_pd = normalization(caracts_pd, min_and_max_pd).to_numpy()

    autoencoder_load = keras.models.load_model(str(model_name))
    metrics  = {"loss": []}

    print("Начало анализа c помощью модели:", model_name)
    valid_metrics_name = ["Расхождение"]
    progress_bar = Progbar(round(len(caracts_pd)/windows_size),
                                 stateful_metrics=valid_metrics_name)

    for idx in range(round(len(caracts_pd)/windows_size)):
        batch_x = np.array([caracts_pd[idx*windows_size:idx*windows_size + windows_size, :]])
        batch_x_restored = autoencoder_load.predict(batch_x, verbose=0)

        loss = keras.losses.mean_squared_error(batch_x, batch_x_restored)
        loss = np.mean(np.array(loss)[0]) * 100
        metrics["loss"].append(loss)
        values = [("Расхождение", loss)]
        progress_bar.add(1, values=values)

    pylab.subplot(1, 1, 1)
    pylab.plot(metrics["loss"])
    pylab.title("loss")

    pylab.show()


if __name__ == '__main__':
    main()