from EventAnalysis import EventAnalyser
from TrafficAnalysis import TrafficAnalyser
from DecisionSystem import DecisionSystem

from contextlib import redirect_stdout
from ipaddress import IPv4Address
from tensorflow import keras
from threading import Thread
from index import interface
from elevate import elevate
from pathlib import Path

import tensorflow as tf
import matplotlib.pyplot as plt
import logging
import time
import json
import sys
import os


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

    # elevate()

    path_net_traffic = "AnomalyDetector\\modeles\\TrafficAnomalyDetector\\1.1\\model_TAD_v1.1"
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
                                       iface_name, ip_client)

    event_analysis = EventAnalyser(buffer_events, NetEvent, protected_devices)

    decision_system = DecisionSystem(buffer_traffic, buffer_events, buffer_output)

    traffic_analysis.start()
    event_analysis.start()
    decision_system.start()

    while True:
        buffer_output_json = json.dumps(buffer_output)

        for record in buffer_output["traffic"]:
            interface.data_cash["traffic"].append(record)
        buffer_output["traffic"].clear()

        for record in buffer_output["events"]:
            interface.data_cash["events"].append(record)
        buffer_output["events"].clear()

        if len(buffer_output["statistics"]) > 0:
            print("\n---------------------------------------")
            print(f"{'RAT_trojans'}: {buffer_output['statistics']['RAT_trojans']}")
            print(f"{'CorrectDetectSafeProcesses'}: {buffer_output['statistics']['CorrectDetectSafeProcesses']}")
            print(f"{'NoCorrectDetectSafeProcesses'}: {buffer_output['statistics']['NoCorrectDetectSafeProcesses']}")
            print(f"{'CorrectDetectRAT_Trojans'}: {buffer_output['statistics']['CorrectDetectRAT_Trojans']}")
            print(f"{'NoCorrectDetectRAT_Trojans'}: {buffer_output['statistics']['NoCorrectDetectRAT_Trojans']}\n")

        interface.SendinData(buffer_output_json)

        time.sleep(1)
