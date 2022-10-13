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
    modeles = "modeles\\model_GRU_traffic\\Checkpoints"
    max_min_file = "modeles\\model_GRU_traffic\\M&M_traffic.csv"
    characts_file = "characts_NoRAT_and_RAT.csv"
    windows_size = 1000

    for model in Path(modeles).iterdir():
        model_name = str(model).split("\\")[-1]

        caracts_pd = pd.read_csv(characts_file)
        caracts_pd = caracts_pd.drop(["Time_Stamp"], axis=1)

        min_and_max_pd = pd.read_csv(max_min_file)
        caracts_pd = normalization(caracts_pd, min_and_max_pd).to_numpy()

        autoencoder_load = keras.models.load_model(str(model))
        metrics  = {"Loss_kld": [], "Loss_mse": [], "Loss_mean_kld": [], "Loss_mean_mse": [], "mae": []}
        mean_kld = keras.metrics.Mean(name="mean_kld")
        mean_mse = keras.metrics.Mean(name="mean_mse")
        mae      = keras.metrics.MeanAbsoluteError(name="mae_mse")

        print("Начало анализа c помощью модели:", model_name)
        for idx in range(round(len(caracts_pd)/windows_size)):
            batch_x = np.array([caracts_pd[idx*windows_size:idx*windows_size + windows_size, :]])
            batch_x_restored = autoencoder_load.predict(batch_x, verbose=0)

            loss_kld = keras.losses.kl_divergence(batch_x, batch_x_restored)
            loss_mse = keras.losses.mse(batch_x, batch_x_restored)

            loss_kld = np.mean(np.array(loss_kld)[0])
            loss_mse = np.mean(np.array(loss_mse)[0])

            mean_kld.update_state(loss_kld)
            mean_mse.update_state(loss_mse)
            mae.update_state(batch_x, batch_x_restored)

            metrics["Loss_kld"].append(loss_kld)
            metrics["Loss_mse"].append(loss_mse)
            metrics["Loss_mean_kld"].append(float(mean_kld.result()))
            metrics["Loss_mean_mse"].append(float(mean_mse.result()))
            metrics["mae"].append((float(mae.result())))

        pd.DataFrame(metrics).to_csv(modeles + "\\" + model_name + "_res.csv", index=False)


    #
    # plt.plot(loss_arr, linewidth=2.0)
    # plt.show()
    #
    #


if __name__ == '__main__':
    main()