from tensorflow import keras
from SnifferPaket.TrafficParcNDPI import PreprocessingPcapng
from pathlib import Path
from SnifferPaket.characts import CHARACTERISTIC
import matplotlib.pyplot as plt
import matplotlib.animation as anim
from datetime import datetime

import pandas as pd
import numpy as np


pcapng_files = [
    "traffic_RAT_NingaliNET_1.pcapng",
    "traffic_RAT_NingaliNET_2.pcapng",
    "traffic_RAT_NingaliNET_3.pcapng",
    "traffic_RAT_NingaliNET_4.pcapng",
    "traffic_RAT_NingaliNET_5.pcapng"
]


def Preproccessing(characts_file, windows_size, path_name):

    path_characts_file = Path(characts_file)
    if path_characts_file.exists() is False:
        pd_ch_name = pd.DataFrame()
        for ch in CHARACTERISTIC:
            pd_ch_name[ch] = []
        pd_ch_name.to_csv(str(path_characts_file), index=False)


    for file in pcapng_files:
        res = PreprocessingPcapng(path_name,
                                  windows_size,
                                  ["192.168.10.128"],
                                  path_name + file,
                                  characts_file)
        if res:
            print("Парсинг выполнен успешно!\n")
        else:
            print("Парсинг не выполнен!\n")
    return 0


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


def plot_cont(yi, xmax):
    y = []
    fig = plt.figure()
    ax = fig.add_subplot(1,1,1)

    def update(i):
        y.append(yi)
        x = range(len(y))
        ax.clear()
        ax.plot(x, y)

    a = anim.FuncAnimation(fig, update, frames=xmax, repeat=False)
    plt.show()


def main():
    model_name = "model_GRU_VNAT_video"
    max_min_file = "max_and_min_e10.csv"
    path_name = "..\\data\\pcap\\temp\\characts.csv"
    characts_file = "characts_NoRAT.csv"
    windows_size = 1000

    # Preproccessing(windows_size)
    # autoencoder = Autoencoder(caracts_count=caracts_count, count_hidden_layers=count_hidden_layers, windows_size=windows_size)
    # autoencoder.compile(optimizer="adam")

    caracts_pd = pd.read_csv(path_name)
    time_s = caracts_pd["Time_Stamp"].to_numpy()
    caracts_pd = caracts_pd.drop(["Time_Stamp"], axis=1)

    min_and_max_pd = pd.read_csv(max_min_file)
    caracts_pd = normalization(caracts_pd, min_and_max_pd).to_numpy()

    autoencoder_load = keras.models.load_model(model_name)
    loss_time = {"Loss": []} # "Time": [],

    print("Начало анализа!")
    print(len(caracts_pd))
    loss_arr = []

    for idx in range(round(len(caracts_pd)/windows_size)):
        batch_x = np.array([caracts_pd[idx*windows_size:idx*windows_size + windows_size, :]])
        batch_x_restored = autoencoder_load.predict(batch_x, verbose=0)
        loss = keras.losses.kl_divergence(batch_x, batch_x_restored)
        mean_loss = np.mean(np.array(loss)[0])
        loss_arr.append(mean_loss)
        loss_time["Loss"].append(mean_loss)

    plt.plot(loss_arr, linewidth=2.0)
    plt.show()

    # print(f"Time: {time},   Loss: {mean_loss}")

    # step = 600
    # for idx in range(0, len(caracts_pd)-windows_size, step):
    #     batch_x = np.array([caracts_pd[idx*step:idx*step + windows_size, :]])
    #     # time_x = time_s[idx:idx + windows_size][0]
    #     # time = datetime.fromtimestamp(time_x).strftime('%H:%M:%S.%f')
    #     batch_x_restored = autoencoder_load.predict(batch_x, verbose=0)
    #     loss = keras.losses.mean_squared_error(batch_x, batch_x_restored)
    #     mean_loss = np.mean(np.array(loss)[0])
    #     # loss_time["Time"].append(time)
    #     loss_time["Loss"].append(mean_loss)
    #     # print(f"Time: {time},   Loss: {mean_loss}")

    pd.DataFrame(loss_time).to_csv("VNAT_res_6.csv", index=False)


if __name__ == '__main__':
    main()