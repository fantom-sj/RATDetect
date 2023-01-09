from EventAnalysis import EventAnalyser
from TrafficAnalysis import TrafficAnalyser
from DecisionSystem import DecisionSystem

from contextlib import redirect_stdout
from ipaddress import IPv4Address
from tensorflow import keras
from threading import Thread
from elevate import elevate
from pathlib import Path

import tensorflow as tf
import matplotlib.pyplot as plt
import logging
import time
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
    # elevate()

    path_net_traffic = "AnomalyDetector\\modeles\\TrafficAnomalyDetector\\0.9.4\\model_TAD_v0.9.4"
    path_net_event   = "AnomalyDetector\\modeles\\EventAnomalyDetector\\0.6.0\\model_EAD_v0.6.0"

    path_traffic_analysis = "WorkDirectory\\"
    protected_devices = {"Жертва": ("192.168.137.1", 62302)}

    iface_name = "VMware_Network_Adapter_VMnet3"
    ip_client = [IPv4Address("192.168.10.128")]

    if not Path("WorkDirectory\\").exists():
        Path("WorkDirectory\\").mkdir()
    if not Path(path_traffic_analysis).exists():
        Path(path_traffic_analysis).mkdir()

    NetTraffic, NetEvent = NeiroNetLoad(path_net_traffic, path_net_event)

    buffer_traffic      = []
    buffer_events       = []
    buffer_anomaly_proc = []

    # traffic_analysis = TrafficAnalyser(buffer_traffic, NetTraffic, path_traffic_analysis,
    #                                    iface_name, ip_client)

    event_analysis = EventAnalyser(buffer_events, NetEvent, protected_devices)

    # decision_system = DecisionSystem(buffer_traffic, buffer_events, buffer_anomaly_proc)

    # traffic_analysis.start()
    event_analysis.start()
    # decision_system.start()

    while True:
        # if len(buffer_anomaly_proc) > 0:
        #     os.system("echo ---------------------------------------")
        #     interim_results = buffer_anomaly_proc.pop(0)
        #     for anomaly in interim_results:
        #
        #         out = f"{anomaly}: {interim_results[anomaly]}"
        #         os.system("echo " + out)

        time.sleep(10)
