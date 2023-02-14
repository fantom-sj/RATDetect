from TrafficAnalysis import TrafficAnalyser
from DecisionSystem import DecisionSystem
from EventAnalysis import EventAnalyser
from AuxiliaryFunctions import logingProcess
from index import interface

from ipaddress import IPv4Address
from tensorflow import keras
from pathlib import Path

import jsonpickle as json
import tensorflow as tf
import logging
import time


def NeiroNetLoad(path_net_traffic, path_net_event):
    check_net_traffic = Path(path_net_traffic).exists()
    check_net_event   = Path(path_net_event).exists()
    if (not check_net_traffic) or (not check_net_event):
        err_net_traffic = f"Указанный путь {path_net_traffic} для нейросети анализа трафика не найден!\n"
        err_net_event   = f"Указанный путь {path_net_traffic} для нейросети анализа процессов системы не найден!\n"
        prompt          = "Укажите верные пути и перезапустите систему анализа."
        if not check_net_traffic: logging.exception(err_net_traffic)
        if not check_net_event:   logging.exception(err_net_event)
        logging.info(prompt)
        exit(-1)
    else:
        with tf.name_scope("NetTraffic") as scope:
            NetTraffic = keras.models.load_model(path_net_traffic)
            print("Нейросеть для анализа сетевого трафика загружена")
        with tf.name_scope("NetEvent") as scope:
            NetEvent   = keras.models.load_model(path_net_event)
            print("Нейросеть для анализа процессов системы загружена")
        return NetTraffic, NetEvent


if __name__ == '__main__':
    interface.start()

    path_net_traffic = "AnomalyDetector\\modeles\\TrafficAnomalyDetector\\1.6.4\\model_TAD_v1.6.4"
    path_net_event   = "AnomalyDetector\\modeles\\EventAnomalyDetector\\0.7.0\\model_EAD_v0.7.0"

    path_traffic_analysis = "WorkDirectory\\"
    protected_devices = {"Жертва": ("192.168.137.1", 62302)}

    iface_name = "VMware_Network_Adapter_VMnet3"
    ip_client = [IPv4Address("192.168.10.128")]

    if not Path("WorkDirectory\\").exists():
        Path("WorkDirectory\\").mkdir()
    if not Path(path_traffic_analysis).exists():
        Path(path_traffic_analysis).mkdir()

    NetTraffic, NetEvent = NeiroNetLoad(path_net_traffic, path_net_event)

    buffer_traffic      = list()
    buffer_events       = list()
    buffer_output       = dict()

    traffic_analysis = TrafficAnalyser(buffer_traffic, NetTraffic, path_traffic_analysis,
                                       iface_name, ip_client, window_size=10)

    event_analysis = EventAnalyser(buffer_events, NetEvent, protected_devices)

    decision_system = DecisionSystem(ip_client, buffer_traffic, buffer_events, buffer_output)

    print("Ожидаем запуск системы мониторинга AntiRAT...")
    while True:
        if interface.process_monitor:
            if not traffic_analysis.is_alive():
                traffic_analysis.run_toggle = True
                try:
                    traffic_analysis.start()
                except RuntimeError:
                    traffic_analysis = TrafficAnalyser(buffer_traffic, NetTraffic, path_traffic_analysis,
                                                       iface_name, ip_client, window_size=10)
                    traffic_analysis.run_toggle = True
                    traffic_analysis.start()

            if not event_analysis.is_alive():
                event_analysis.run_toggle   = True
                try:
                    event_analysis.start()
                except RuntimeError:
                    event_analysis = EventAnalyser(buffer_events, NetEvent, protected_devices)
                    event_analysis.run_toggle = True
                    event_analysis.start()

            if not decision_system.is_alive():
                decision_system.run_toggle  = True
                try:
                    decision_system.start()
                except RuntimeError:
                    decision_system = DecisionSystem(ip_client, buffer_traffic, buffer_events, buffer_output)
                    decision_system.run_toggle = True
                    decision_system.start()

        else:
            if traffic_analysis.is_alive():
                traffic_analysis.run_toggle = False
            if event_analysis.is_alive():
                event_analysis.run_toggle   = False
            if decision_system.is_alive():
                decision_system.run_toggle  = False
                print("Процесс системы мониторинга AntiRAT приостановлен!")
            time.sleep(0.5)
            continue

        if interface.type_data == "data_analys_res":
            buffer_json = json.dumps(buffer_output["AnalysisResults"])
        elif interface.type_data == "data_events":
            buffer_json = json.dumps(buffer_output["events"])
        elif interface.type_data == "data_netflow":
            buffer_json = json.dumps(buffer_output["traffic"])
        else:
            buffer_json = None

        interface.SendinData(buffer_json)
        logingProcess(buffer_output["events"], buffer_output["traffic"])

        for record in buffer_output["traffic"]:
            interface.data_cash["traffic"].append(record)
        buffer_output["traffic"].clear()

        for record in buffer_output["events"]:
            interface.data_cash["events"].append(record)
        buffer_output["events"].clear()

        interface.data_cash["analiz_res"] = buffer_output["AnalysisResults"]

        print("\n\n----------------------------------------")
        print(f"Правильно определённые безопасные процессы: "
              f"{buffer_output['statistics']['CorrectDetectSafeProcesses']}")
        print(f"Ошибочно определённые безопасные процессы: "
              f"{buffer_output['statistics']['NoCorrectDetectSafeProcesses']}")
        print(f"Правильно определённые RAT-трояны: "
              f"{buffer_output['statistics']['CorrectDetectRAT_Trojans']}")
        print(f"Ошибочно определённые безопасные RAT-трояны: "
              f"{buffer_output['statistics']['NoCorrectDetectRAT_Trojans']}")
        print("----------------------------------------\n")

        time.sleep(3)