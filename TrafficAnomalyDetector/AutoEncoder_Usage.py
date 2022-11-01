import tensorflow as tf
from tensorflow import keras
from keras.utils import Progbar

from AutoEncoder_RNN import Autoencoder, TrainingDatasetGen

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from scipy.signal import savgol_filter


def main():
    # time.sleep(7500)

    # Парамеры автоэнкодера
    versia          = "0.8.5.4.0"
    arhiteche       = {"GRU_1": 13, "GRU_2": 12, "GRU_3": 11, "GRU_4": 10, "GRU_5": 11, "GRU_6": 12, "GRU_7": 13}

    epochs_count    = 5
    batch_size      = 1000
    windows_size    = 1000
    loss_func       = keras.losses.mse
    max_min_file    = "modeles\\TrafficAnomalyDetector\\" + versia + "\\M&M_traffic_VNAT.csv"
    model_check     = "modeles\\TrafficAnomalyDetector\\" + versia + "\\Checkpoint\\itter_"

    # Тестовый датасет
    characts_file   = "..\\data\\pcap\\test_dataset\\array_characts.csv"
    feature_range   = (-1, 1)
    caracts_data    = pd.read_csv(characts_file)
    caracts_pd      = caracts_data.drop(["Time_Stamp"], axis=1)
    caracts_pd      = caracts_pd.drop(["Dev_size_TCP_paket"], axis=1)
    caracts_pd      = caracts_pd.drop(["Dev_size_UDP_paket"], axis=1)
    caracts_pd      = caracts_pd.drop(["Dev_client_paket_size"], axis=1)
    caracts_pd      = caracts_pd.drop(["Dev_server_paket_size"], axis=1)
    caracts_pd      = caracts_pd.drop(["RAT_count"], axis=1)
    caracts_numpy   = TrainingDatasetGen.normalization(caracts_pd, max_min_file, feature_range).to_numpy()

    numbs_count, caracts_count  = caracts_numpy.shape
    batch_count                 = round(numbs_count/windows_size) - 1
    caracts_numpy               = caracts_numpy[:batch_count*windows_size]
    tensor_dataset              = tf.convert_to_tensor(caracts_numpy)
    tensor_dataset              = tf.reshape(tensor_dataset, (batch_count, windows_size, caracts_count))

    # Определение автоэнкодера
    autoencoder = Autoencoder(caracts_count, arhiteche, batch_size=batch_size, windows_size=windows_size)
    autoencoder.build((1, windows_size, caracts_count))
    autoencoder.summary()
    autoencoder.compile(optimizer="adam", loss=loss_func)
    print("Автоэнкодер определён.")

    print("Начинаем прогнозирование аномального трафика.")
    metrics_analiz = {}
    for epoch in range(1, epochs_count+1, 1):
        print(f"Тестируем модель: {versia}, эпоха: {epoch}")

        metrics_analiz["loss_epoch"+str(epoch)] = []
        autoencoder.load_weights(model_check + str(epoch))

        valid_metrics_name = ["Расхождение"]
        progress_bar = Progbar(batch_count, stateful_metrics=valid_metrics_name)

        for idx in range(batch_count):
            batch_x = tf.gather(tensor_dataset, idx, axis=0)
            batch_x = tf.reshape(batch_x, (1, windows_size, caracts_count))
            batch_x_restored = autoencoder.predict(batch_x, verbose=0)

            loss = loss_func(batch_x, batch_x_restored)
            loss = np.mean(np.array(loss)[0]) * 100
            metrics_analiz["loss_epoch"+str(epoch)].append(loss)
            values = [("Расхождение", loss)]
            progress_bar.add(1, values=values)

        metrics_analiz["loss_epoch"+str(epoch)] = savgol_filter(metrics_analiz["loss_epoch"+str(epoch)], 31, 3)

    print("Посчитываем реальный аномальный трафик.")
    array_characts = caracts_data.to_dict("records")[:batch_count*windows_size]
    metrics_analiz["real_loss"] = []
    for idx in range(0, batch_count*windows_size, windows_size):
        ch_srez = array_characts[idx:idx + windows_size]
        RAT_count_sum = 0
        for ch in ch_srez:
            RAT_count_sum += int(ch["RAT_count"])
        RAT_count_mean = RAT_count_sum / windows_size
        metrics_analiz["real_loss"].append(RAT_count_mean)
    metrics_analiz["real_loss"] = savgol_filter(metrics_analiz["real_loss"], 31, 3)

    metrics_analiz_pd = TrainingDatasetGen.normalization(pd.DataFrame(metrics_analiz), feature_range=(0, 100))
    metrics_analiz_norm = metrics_analiz_pd.to_dict("list")

    mng = plt.get_current_fig_manager()
    mng.window.showMaximized()

    for epoch in range(1, epochs_count + 1, 1):
        plt.xlim([-5.0, 1210.0])
        plt.ylim([-5.0, 105.0])
        plt.title(f"Ошибки восстановления на эпохе {epoch}, версия: {versia}")
        plt.grid(which='major')
        plt.grid(which='minor', linestyle=':')

        plt.plot(metrics_analiz_norm["real_loss"], label="Реальные аномалии", color="tab:red")
        plt.plot(metrics_analiz_norm["loss_epoch" + str(epoch)], label="Обнаруженные аномалии", color="tab:blue")

        plt.legend(fontsize=10)
        plt.tight_layout()
        plt.show()


if __name__ == '__main__':
    main()