import pickle
import pandas as ps
from collections import Counter

from modules.ProcessPreProcessing import ProcPreProcessOne

ModelForTraffic = "mdl/madel_traffic.mdl"
ModelForProcess = "mdl/madel_process.mdl"

model_traffic, scaler_traffic, labeler_traffic \
    = pickle.load(open(ModelForTraffic, "rb"))

model_process, scaler_process, labeler_process \
    = pickle.load(open(ModelForProcess, "rb"))


def EventAnalysis(model, scaler, labeler, data):
    X = scaler.transform(data)
    y = model.predict(X)
    labels = labeler.inverse_transform(y)
    count = Counter(labels)
    noRat = count["NoRAT"]
    Rat   = count["RAT"]
    return (round(noRat / len(labels) * 1000)/1000), (round(Rat / len(labels) * 1000)/1000)


def CsvProcessing(dump_csv):
    white_list = []
    gray_list =  []
    black_list = []

    proc_log = []
    proc_pid = ""
    str_num = 0
    log_data = ps.read_csv(dump_csv)
    for index, row in log_data.iterrows():
        if (str_num == 0) and \
                (row["PID"] not in white_list) and \
                (row["PID"] not in gray_list) and \
                (row["PID"] not in black_list):
            proc_pid = row["PID"]
            proc_log.append({
                "Время суток": row["Время суток"],
                "Операция": row["Операция"],
                "Путь": row["Путь"],
                "Результат": row["Результат"],
                "Подробности": row["Подробности"]
            })
            str_num += 1
        elif 0 < str_num < 100:
            if row["PID"] == proc_pid:
                proc_log.append({
                    "Время суток": row["Время суток"],
                    "Операция": row["Операция"],
                    "Путь": row["Путь"],
                    "Результат": row["Результат"],
                    "Подробности": row["Подробности"]
                })
                str_num += 1
            else:
                continue

        elif str_num == 100:
            str_num = 0
            data_process = ProcPreProcessOne(proc_log)

            res = EventAnalysis(model_process, scaler_process, labeler_process, data_process)
            print(res)









if __name__ == '__main__':
    main()

    # '.\Procmon64.exe /BackingFile "Log.pml" /Runtime 10'