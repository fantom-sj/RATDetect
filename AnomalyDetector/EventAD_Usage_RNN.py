from AutoEncoder_RNN import TrainingDatasetGen
from matplotlib.widgets import Slider
import matplotlib.pyplot as plt
from keras.utils import Progbar
from tensorflow import keras
from tqdm import tqdm

import tensorflow as tf
import pandas as pd
import numpy as np
import logging
import math


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

    print("Реальные аномалии в тестовом наборе выявлены")
    return real_anomaly


def PostAnalys(caracts_np, metrics_analiz):
    RAT = {"RAT_client.exe", "1.exe", "RAT_client_Rev.exe"}

    count_RAT = 0
    count_no_RAT = 0
    for ch in caracts_np["Process_Name"]:
        if ch in RAT:
            count_RAT += 1
        else:
            count_no_RAT += 1

    false_positive = 0
    false_negative = 0
    true_positive = 0
    true_negative = 0

    for i in range(len(metrics_analiz["loss"])):
        if metrics_analiz["loss"][i] > 0.0821:
            if caracts_np["Process_Name"][i] in RAT:
                true_positive += 1
            else:
                false_positive += 1
        else:
            if caracts_np["Process_Name"][i] in RAT:
                false_negative += 1
            else:
                true_negative += 1

    print(f"Всего RAT-троянов:             {count_RAT}")
    print(f"Всего нормальных событий:      {count_no_RAT}")
    print(f"Правильно-позитивная реакция:  {true_positive}")
    print(f"Правильно-негативная реакция:  {true_negative}")
    print(f"Ложно-позитивная реакция:      {false_positive}")
    print(f"Ложно-негативная реакция:      {false_negative}")

    plt.title(f"Результаты работы нейросети")
    plt.grid(which='major')
    plt.grid(which='minor', linestyle=':')

    # fig, ax = plt.subplots()
    ax_slide = plt.axes([0.25, 0.01, 0.65, 0.05])
    porog = Slider(ax_slide, "Порог", valmin=0.01, valmax=0.1, valinit=0.0821, valstep=0.001)

    def update(val):
        global bars
        false_positive = 0
        false_negative = 0
        true_positive = 0
        true_negative = 0

        for i in range(len(metrics_analiz["loss"])):
            if metrics_analiz["loss"][i] > val:
                if caracts_np["Process_Name"][i] in RAT:
                    true_positive += 1
                else:
                    false_positive += 1
            else:
                if caracts_np["Process_Name"][i] in RAT:
                    false_negative += 1
                else:
                    true_negative += 1

        print(f"Правильно-позитивная реакция:  {true_positive}")
        print(f"Правильно-негативная реакция:  {true_negative}")
        print(f"Ложно-позитивная реакция:      {false_positive}")
        print(f"Ложно-негативная реакция:      {false_negative}")

    plt.tight_layout()
    porog.on_changed(update)
    plt.show()


def main():
    versia = "0.6.0"

    # Парамеры автоэнкодера
    batch_size      = 1
    window_size     = 1
    loss_func       = keras.losses.mse
    path_model      = "modeles\\EventAnomalyDetector\\" + versia + "\\"
    max_min_file    = path_model + "M&M_event.csv"
    model           = path_model + "model_EAD_v" + versia
    results         = path_model + "res_"
    porog_anomaly   = 0.3

    anomaly_proc    = {"NingaliNET": "RAT_client", "Rabbit-Hole": "1.exe", "Revenge-RAT": "RAT_client_Rev"}
    color_RAT       = {"NingaliNET": "tab:red", "Rabbit-Hole": "tab:green", "Revenge-RAT": "tab:purple"}
    count_epoch     = 3

    path_name         = "\\\\VictimPC\\RATDetect\\WorkDirectory"
    charact_file_mask = "events_characters_"

    characts_pd = pd.DataFrame()
    for charact_index in range(112):
        if charact_index == 76 or charact_index == 83:
            continue
        characts_pd = pd.concat([characts_pd, pd.read_csv(f"{path_name}\\{charact_file_mask}{charact_index}.csv")],
                                ignore_index=True)
    characts_pd.sort_values(by="Events_Charact.Time_Stamp_End")

    characts_pd      = characts_pd[characts_pd["Events_Charact.Process_Name"] != "python.exe"]
    feature_range    = (-1, 1)
    characts_np      = characts_pd.to_dict("records")
    characts_pd      = characts_pd.drop(["Events_Charact.Time_Stamp_Start"], axis=1)
    characts_pd      = characts_pd.drop(["Events_Charact.Time_Stamp_End"], axis=1)
    characts_pd      = characts_pd.drop(["Events_Charact.Process_Name"], axis=1)
    characts_pd      = characts_pd.drop(["Events_Charact.Direction_IP_Port"], axis=1)
    characts_pd      = characts_pd.drop(["Events_Charact.Count_Events_Batch"], axis=1)
    characts_pd      = characts_pd.drop(["Events_Charact.Duration"], axis=1)
    caracts_numpy    = TrainingDatasetGen.normalization(characts_pd, max_min_file, feature_range, True)

    # print(f"Анализируем тестовый набор на реальные аномалии")
    # real_anomaly = GetRealAnomaly(anomaly_proc, window_size, caracts_np, color_RAT)
    # pd_real_anomaly = pd.DataFrame(real_anomaly)
    # pd_real_anomaly.to_csv(results+"real_anomaly.csv", index=False)

    numbs_count, caracts_count    = caracts_numpy.shape
    batch_count                   = math.floor(numbs_count/batch_size)

    # Определение автоэнкодера
    autoencoder = tf.keras.models.load_model(model)
    # autoencoder.load_weights(model+str(model_index))
    print(f"Модель {model} загружена")

    print("Начинаем прогнозирование аномальных событий.")
    metrics_analiz = {}

    valid_metrics_name = ["Расхождение"]
    # progress_bar = Progbar(batch_count, stateful_metrics=valid_metrics_name)

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
            mean_loss = tf.math.reduce_mean(loss) # tf.math.multiply(tf.math.reduce_mean(loss), tf.constant(1, dtype=tf.float32))
            values = [("Расхождение", mean_loss)]
            # progress_bar.add(1, values=values)

        except Exception as err:
            logging.exception(f"Ошибка!\n{err}")
            print(np.array(batch_x).shape)
            continue

    process_anomal_res = {}
    data = np.array(metrics_analiz["loss"])

    for idx in range(len(characts_np)):
        process_name = characts_np[idx]["Events_Charact.Process_Name"]
        if not process_name in process_anomal_res:
            process_anomal_res[process_name] = list()
        process_anomal_res[process_name].append(data[idx])

    anomaly_level_process = {}
    for process in process_anomal_res:
        anomaly_level_process[process] = 0
        for loss in process_anomal_res[process]:
            if loss >= porog_anomaly:
                anomaly_level_process[process] += 1

    no_anomaly_level_process = {}
    for process in process_anomal_res:
        no_anomaly_level_process[process] = 0
        for loss in process_anomal_res[process]:
            if loss < porog_anomaly:
                no_anomaly_level_process[process] += 1

    print("\nУровень аномальной активности для каждого процесса:")
    for process in anomaly_level_process:
        print(f"{process}: {anomaly_level_process[process]}")

    print("\nУровень нормальной активности для каждого процесса:")
    for process in no_anomaly_level_process:
        print(f"{process}: {no_anomaly_level_process[process]}")

    mng = plt.get_current_fig_manager()
    mng.window.showMaximized()

    len_metrix = len(metrics_analiz["loss"])
    plt.xlim([-5.0, len_metrix + 5])
    plt.ylim([-0.02, 1.01])
    plt.title(f"График аномалий в событиях процессов в ОС")
    plt.grid(which='major')
    plt.grid(which='minor', linestyle=':')

    plt.plot([porog_anomaly for _ in range(len_metrix)], label="Уровень нормальных данных", color="tab:red")
    plt.plot(metrics_analiz["loss"], label="Обнаруженные аномалии", color="tab:blue")

    plt.legend(fontsize=10)
    plt.tight_layout()
    plt.show()

    # pd_metric_on_model = pd.DataFrame(process_anomal_res)
    # pd_metric_on_model.to_csv(results + "prognoses_anomaly.csv", index=False)


if __name__ == '__main__':
    main()