import os
from pathlib import Path
import pandas as pd


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

    csv_all.to_csv(path + "itog_video.csv", index=False)


def main():
    path = Path("F:\\VNAT\\temp")
    file_arr = []
    for file in path.iterdir():
        file = str(file)
        if (".csv" in file and "nonvpn" in file) and \
                ("netflix" in file or "vimeo" in file or "youtube" in file):
            file_arr.append(str(file))
            print(str(file))

    merge_csv(str(path), file_arr)


if __name__ == '__main__':
    main()
