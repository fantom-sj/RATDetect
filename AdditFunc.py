import os
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter
from TrafficAnomalyDetector.AutoEncoder_RNN import TrainingDatasetGen

def SearchNearest(num, arr):
    max = 0
    id_max = 0
    id_min = 0
    for i in range(len(arr)):
        if (arr[i] < num) and (arr[i] > max):
            max = arr[i]
            id_max = i

    min = 1000
    for i in range(len(arr)):
        if (arr[i] > num) and (arr[i] < min):
            min = arr[i]
            id_min = i

    if abs(num - max) < abs(num - min):
        return id_max
    elif abs(num - max) == abs(num - min):
        return id_max, id_min
    else :
        return id_min


def merge_csv(path, csv_file_arr):
    """
        Функция для слияния заданного списка csv файлов, с последующим сохранением в итоговый csv
    """
    print("Начато объединение файлов.")
    csv_all = pd.DataFrame()
    for file in csv_file_arr:
        temp_pandas = pd.read_csv(file)
        csv_all = pd.concat([csv_all, temp_pandas], ignore_index=False)

    csv_all.to_csv(path, index=False)


def print_model_res(file_path, train, valid, feature_range = (0, 100),
                    window_length=31, window_length_valid=3, polyorder=3):
    path = Path(file_path)
    arr_file = []
    if path.exists():
        for file in path.iterdir():
            if train and "train" in str(file).split("\\")[-1]:
                arr_file.append(str(file))
            elif valid and "valid" in str(file).split("\\")[-1]:
                arr_file.append(str(file))
            else:
                continue
    else:
        print(f"Ошибка: Директория {file_path} не найдена.")

    pd_data_arr = {}
    if len(arr_file)>0:
        for file in arr_file:
            file_name = str(file).split("\\")[-1]
            print(f"Эагружаем файл: {file_name}")
            pd_data_arr[file_name] = pd.read_csv(file)
            print(f"Файл {file_name} успешно загружен.\n\n")
    else:
        print(f"Ошибка: В директории {file_path} нет файлов удовлетворяющих заданным требованиям.")

    window_length_orign = window_length
    for file in pd_data_arr:
        if "train_e" in file:
            epoch       = file.split("_")[2][1]
            model       = file.split("_")[3].split(".c")[0]
            name_graf   = f"История обучения в эпоху {epoch} для модели {model}"
        elif "train" in file:
            model       = file.split("_")[2].split(".c")[0]
            name_graf   = f"История обучения модели"
        elif "valid_e" in file:
            epoch       = file.split("_")[2][1]
            model       = file.split("_")[3].split(".c")[0]
            name_graf   = f"История валидации в эпоху {epoch} для модели {model}"
            window_length = window_length_valid
            polyorder = 1
        elif "valid" in file:
            model       = file.split("_")[2].split(".c")[0]
            name_graf   = f"История всей валидации модели {model}"
            window_length = window_length_valid
            polyorder = 1
        else:
            name_graf   = "Неизвестные данные"

        loss        = pd_data_arr[file]["loss"].to_numpy()
        mean_loss   = pd_data_arr[file]["mean_loss"].to_numpy()
        mae         = pd_data_arr[file]["mae"].to_numpy()

        if len(loss) == 0 or len(mean_loss) == 0 or len(mae) == 0:
            print(f"В файле {file} нет данных об одной из трёх метрик!\n")
            continue
        elif window_length > len(loss):
            print(f"Для файла {file} задана не подходящая длина окна!\n" 
                  f"В данном файле метрики содержат по {len(loss)} записей,\n"
                  f"введите размер меньший или равный этому числу, а также большую чем {polyorder}:")
            window_length = int(input())

        print(f"Сглаживание метрик из файла: {file}")
        loss        = savgol_filter(loss, window_length, polyorder)
        mean_loss   = savgol_filter(mean_loss, window_length, polyorder)
        mae         = savgol_filter(mae, window_length, polyorder)

        print(f"Нормализация метрик из файла: {file}\n")
        pd_metrics      = pd.DataFrame({"loss": loss, "mean_loss": mean_loss, "mae": mae})
        norm_metrics    = TrainingDatasetGen.normalization(pd_metrics, feature_range=feature_range)

        mng = plt.get_current_fig_manager()
        mng.window.showMaximized()

        plt.xlim([-5.0, len(loss)])
        plt.ylim([-5.0, 105.0])
        plt.title(name_graf)
        plt.grid(which='major')
        plt.grid(which='minor', linestyle=':')

        plt.plot(norm_metrics["loss"], label="Ошибка прогнозирования", color="tab:red")
        plt.plot(norm_metrics["mean_loss"], label="Средний уровень ошибок", color="tab:blue")
        # plt.plot(norm_metrics["mae"], label="Средняя абсолютная ошибка", color="tab:green")

        plt.legend(fontsize=10)
        plt.tight_layout()
        plt.show()


def main():
    # path = Path("F:\\VNAT\\temp")
    # file_arr = []
    # for file in path.iterdir():
    #     file = str(file)
    #     if ".csv" in file and "nonvpn" in file:
    #         file_arr.append(str(file))
    #         print(str(file))
    #
    # merge_csv(str(path) + "\\VNAT_nonvpn.csv", file_arr)

    # merge_csv("F:\\VNAT\\Mytraffic\\youtube_me\\learn_and_valid_dataset\\dataset_all.csv", [
    #     "F:\\VNAT\\Mytraffic\\youtube_me\\learn_and_valid_dataset\\dataset_0.csv",
    #     "F:\\VNAT\\Mytraffic\\youtube_me\\learn_and_valid_dataset\\dataset_1.csv",
    #     "F:\\VNAT\\Mytraffic\\youtube_me\\learn_and_valid_dataset\\dataset_2.csv",
    #     "F:\\VNAT\\Mytraffic\\youtube_me\\learn_and_valid_dataset\\dataset_3.csv",
    #     "F:\\VNAT\\Mytraffic\\youtube_me\\learn_and_valid_dataset\\dataset_4.csv",
    #     "F:\\VNAT\\Mytraffic\\youtube_me\\learn_and_valid_dataset\\dataset_5.csv"
    # ])

    versia          = "0.8.6.2"
    path_model      = "D:\\Пользователи\\Admin\\Рабочий стол\\Статья по КБ\\RATDetect\\" \
                      "TrafficAnomalyDetector\\modeles\\TrafficAnomalyDetector\\" + versia
    train           = True
    valid           = False
    feature_range   = (0, 100)
    window_length   = 31
    window_valid    = 2
    polyorder       = 3

    print_model_res(path_model, train, valid, feature_range, window_length, window_valid, polyorder)


if __name__ == '__main__':
    main()
