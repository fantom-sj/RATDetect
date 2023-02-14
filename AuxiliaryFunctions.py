from pathlib import Path
import pandas as pd


def GetFilesCSV(path):
    path = Path(path)
    files_local = {}
    files_timecreate = []

    for file in path.iterdir():
        file_split = str(file).split(".")

        if file.is_file():
            if file_split[1] == "csv" or file_split[1] == "CSV":
                size_file = file.stat().st_size
                if size_file > 0:
                    try:
                        old_name_file = str(file)
                        new_name_file = old_name_file + "_tmp"
                        file.rename(new_name_file)
                        Path(new_name_file).rename(old_name_file)
                        time_create = file.stat().st_mtime
                        files_timecreate.append(time_create)
                        files_local[time_create] = str(file)
                    except PermissionError:
                        continue
                else:
                    continue
            else:
                continue
        else:
            continue

    files_characts_arr = []
    if len(files_timecreate) > 0:
        files_timecreate.sort()

        for tc in files_timecreate:
            files_characts_arr.append(files_local[tc])
        return files_characts_arr
    else:
        return []


def createLogFile(path_log: str, log_buffer: list, type_log: str):
    if type_log == "event":
        logs_data = pd.DataFrame(columns=["ID", "Time_Stamp_Start", "Time_Stamp_End",
                                          "Process_Name", "connection", "loss", "anomaly"])
    else:
        logs_data = pd.DataFrame(columns=["ID", "Time_Stamp_Start", "Time_Stamp_End", "Src_IP_Flow",
                                          "Dst_IP_Flow", "Src_Port_Flow", "Dst_Port_Flow", "loss", "anomaly"])

    files_logs = GetFilesCSV(path_log)
    if len(files_logs) == 0:
        log_index = -1
    else:
        log_index = int(files_logs[-1].split("\\")[-1].split(".")[0].split("_")[-1])

    logs_data = pd.concat([logs_data, pd.DataFrame(log_buffer)], ignore_index=True)

    if log_index == -1:
        log_index += 1
        file_logs_name = path_log + "\\log_" + type_log + "_" + str(log_index) + ".csv"
        logs_data.to_csv(file_logs_name, index=False)
    else:
        file_logs_name = path_log + "\\log_" + type_log + "_" + str(log_index) + ".csv"
        old_logs = pd.read_csv(file_logs_name)
        if len(old_logs) > 500:
            log_index += 1
            file_logs_name = path_log + "\\log_" + type_log + "_" + str(log_index) + ".csv"
            logs_data.to_csv(file_logs_name, index=False)
        else:
            pd.concat([old_logs, logs_data], ignore_index=True).to_csv(file_logs_name, index=False)


def logingProcess(buffer_events, buffer_netflows):
    path_events_log   = "LogsHistory\\LogEvents"
    path_netflows_log = "LogsHistory\\LogNetFlows"

    createLogFile(path_events_log, buffer_events, "event")
    createLogFile(path_netflows_log, buffer_netflows, "netflows")