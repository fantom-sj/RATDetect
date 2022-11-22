from ProcessEventAnalis.PMLParser import ParserEvents
from AutoEncoder_RNN import TrainingDatasetGen, Autoencoder
from scipy.signal import savgol_filter
from keras.utils import Progbar
from tensorflow import keras
from tqdm import tqdm

import matplotlib.pyplot as plt
import tensorflow as tf
import pandas as pd
import numpy as np
import logging


def GetRealAnomaly(anomaly_proc, window_size, caracts_np, color_RAT):
    real_anomaly = {}
    for RAT in anomaly_proc:
        real_anomaly[RAT] = []
        pbar = tqdm(total=len(caracts_np)-window_size, desc=f"Анализ для процесса {RAT}")
        for i in range(window_size, len(caracts_np), 1):
            count_RAT = 0
            for ch in caracts_np[i - window_size:i]:
                if anomaly_proc[RAT] in ch[2]:
                    count_RAT += 1
            real_anomaly[RAT].append(count_RAT)
            pbar.update(1)
        pbar.close()

        print(f"Нормализация для процесса: {RAT}...")
        inten_max = max(real_anomaly[RAT])
        for i in range(len(real_anomaly[RAT])):
            real_anomaly[RAT][i] = real_anomaly[RAT][i] / inten_max * 100

    plt.xlim([-5.0, len(caracts_np) + 5])
    plt.ylim([-5.0, 105.0])
    plt.title(f"График интенсивности аномальных событий процессов")
    plt.grid(which='major')
    plt.grid(which='minor', linestyle=':')

    for RAT in real_anomaly:
        plt.plot(real_anomaly[RAT], label=RAT, color=color_RAT[RAT])

    plt.legend(fontsize=10)
    plt.tight_layout()
    plt.show()

    print("Реальные аномалии в тестовом наборе выявлены")
    return real_anomaly


def main(versia):
    # Парамеры автоэнкодера
    batch_size      = 800
    window_size     = 100
    loss_func       = keras.losses.mse
    max_min_file    = "modeles\\EventAnomalyDetector\\" + versia + "\\M&M_event.csv"
    model           = "modeles\\EventAnomalyDetector\\" + versia + "\\Checkpoint\\epoch_"
    results         = "modeles\\EventAnomalyDetector\\" + versia + "\\res_"

    anomaly_proc  = {"NingaliNET": "RAT_client", "Rabbit-Hole": "1.exe", "Revenge-RAT": "RAT_client_Rev"}
    color_RAT     = {"NingaliNET": "tab:red", "Rabbit-Hole": "tab:green", "Revenge-RAT": "tab:purple"}

    characts_file   = "F:\\EVENT\\EventTest\\test_112_dataset_0.csv"
    feature_range   = (-1, 1)
    caracts_pd      = pd.read_csv(characts_file)
    caracts_np      = caracts_pd.to_numpy()[300000:]
    caracts_pd      = caracts_pd.drop(["Time_Stamp_Start"], axis=1)
    caracts_pd      = caracts_pd.drop(["Time_Stamp_End"], axis=1)
    caracts_pd      = caracts_pd.drop(["Process_Name"], axis=1)
    caracts_pd      = caracts_pd.drop(["Count_Events_Batch"], axis=1)
    caracts_pd      = caracts_pd.drop(["Count_System_Statistics"], axis=1)
    caracts_numpy   = TrainingDatasetGen.normalization(caracts_pd, max_min_file, feature_range, True).to_numpy()[300000:]
    if np.isnan(np.sum(caracts_numpy)):
        print("В тестовом датасете были обнаружены nan-данные, они были заменены на 0")
        caracts_numpy = np.nan_to_num(caracts_numpy)

    # print(f"Анализируем тестовый набора на реальные аномалии")
    # real_anomaly = GetRealAnomaly(anomaly_proc, window_size, caracts_np, color_RAT)
    # pd_real_anomaly = pd.DataFrame(real_anomaly)
    # pd_real_anomaly.to_csv(results+"real_anomaly.csv", index=False)

    numbs_count, caracts_count    = caracts_numpy.shape
    batch_count                   = round(numbs_count/batch_size) - 11

    metric_on_model = {}
    # autoencoder = Autoencoder(caracts_count, arhiteche, window_size)

    for model_index in range(1, 6, 1):
        # Определение автоэнкодера
        autoencoder = tf.keras.models.load_model(model+str(model_index))
        # autoencoder.load_weights(model+str(model_index))
        print(f"Модель {model+str(model_index)} загружена")

        print("Начинаем прогнозирование аномальных событий.")
        metrics_analiz = {}

        valid_metrics_name = ["Расхождение"]
        progress_bar = Progbar(batch_count, stateful_metrics=valid_metrics_name)

        for idx in range(0, batch_count, 1):
            batch_x = []
            for i in range(batch_size):
                batch_x.append(caracts_numpy[i + (idx * batch_size):window_size + i + (idx * batch_size)])
            try:
                batch_x = tf.convert_to_tensor(batch_x)
                batch_x_restored = autoencoder.__call__(batch_x)

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

            # metrics_analiz["loss"] = \
            #     savgol_filter(np.array(metrics_analiz["loss"]), 31, 3)

        # metrics_analiz_pd = TrainingDatasetGen.normalization(pd.DataFrame(metrics_analiz),
        #                                                      feature_range=(0, 100), mix_max_from_file=False)
        # metrics_analiz_norm = metrics_analiz_pd.to_dict("list")
        metric_on_model[f"epocha_{model_index}"] = metrics_analiz["loss"]

        # mng = plt.get_current_fig_manager()
        # mng.window.showMaximized()
        #
        # len_metrix = len(metrics_analiz_norm["loss"])
        # plt.xlim([-5.0, len_metrix + 5])
        # plt.ylim([-5.0, 105.0])
        # plt.title(f"График аномалий в потоке событие процессов")
        # plt.grid(which='major')
        # plt.grid(which='minor', linestyle=':')
        #
        # for RAT in real_anomaly:
        #     plt.plot(real_anomaly[RAT], label=RAT, color=color_RAT[RAT])
        # plt.plot(metrics_analiz_norm["loss"], label="Обнаруженные аномалии", color="tab:blue")
        #
        # plt.legend(fontsize=10)
        # plt.tight_layout()
        # plt.show()

    pd_metric_on_model = pd.DataFrame(metric_on_model)
    pd_metric_on_model.to_csv(results + "prognoses_anomaly.csv", index=False)

if __name__ == '__main__':
    main("0.4.1_GRU")
    main("0.4.2_LSTM")