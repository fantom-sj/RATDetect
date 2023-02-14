from threading import Thread
from pathlib import Path
import eel
import pandas as pd


class Interface(Thread):
    def __init__(self):
        super().__init__()

        self.data_cash = dict()
        self.data_cash["traffic"] = list()
        self.data_cash["events"] = list()
        self.data_cash["analiz_res"] = None

        self.max_records = 100
        self.type_data   = None

        self.process_monitor = False

    def run(self):
        eel.init("Interface")
        eel.start("index.html", mode="edge")

    def SendinData(self, buffer_json):
        # print("Выводим данные в форму")
        if (self.type_data is not None) and (buffer_json is not None):
            eel.receiver(buffer_json, self.type_data)


interface = Interface()


@eel.expose
def SendDataCash(type_data):
    interface.type_data = type_data

    if type_data == "data_events" or type_data == "data_netflow":
        eel.SetMaxRecords(interface.max_records)
        if type_data == "data_events" and len(interface.data_cash["events"]) > 0:
            if interface.max_records == 0:
                eel.receiverDataCash(type_data, interface.data_cash["events"])
            else:
                eel.receiverDataCash(type_data, interface.data_cash["events"][-interface.max_records:])
        elif type_data == "data_netflow" and len(interface.data_cash["traffic"]) > 0:
            if interface.max_records == 0:
                eel.receiverDataCash(type_data, interface.data_cash["traffic"])
            else:
                eel.receiverDataCash(type_data, interface.data_cash["traffic"][-interface.max_records:])
    elif type_data == "data_analys_res":
        eel.receiverMonitStat(interface.process_monitor)
        if interface.data_cash["analiz_res"] is not None:
            eel.receiver(interface.data_cash["analiz_res"], type_data)


@eel.expose
def SendMaxRecords(max_records):
    if max_records == "all":
        interface.max_records = 0
    else:
        try:
            interface.max_records = int(max_records)
        except:
            interface.max_records = 0


@eel.expose
def MonitorToggle(switch_tog):
    if switch_tog:
        interface.process_monitor = True
    else:
        interface.process_monitor = False


@eel.expose
def get_files_log(type_log):
    path_events_log = "LogsHistory\\LogEvents"
    path_netflows_log = "LogsHistory\\LogNetFlows"

    if type_log == "events":
        path_log = path_events_log
    else:
        path_log = path_netflows_log

    files_log = []
    for file in Path(path_log).iterdir():
        files_log.append(str(file).split("\\")[-1])

    eel.set_files_log(files_log, type_log)


@eel.expose
def get_log_file(type_log, file_name):
    path_events_log = "LogsHistory\\LogEvents\\"
    path_netflows_log = "LogsHistory\\LogNetFlows\\"

    if str(file_name) == "default":
        return
    print(type_log)
    if type_log == "events":
        print(111)
        path_log_file = path_events_log + str(file_name)
    else:
        path_log_file = path_netflows_log + str(file_name)

    log_data = pd.read_csv(path_log_file).to_dict("records")

    eel.set_log_data(log_data, type_log)
