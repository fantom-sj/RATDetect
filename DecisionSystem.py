import logging
import time
from threading import Thread
from pathlib import Path
from ipaddress import IPv4Address
import os
import numpy as np

class DecisionSystem(Thread):
    def __init__(self, buffer_traffic: list, buffer_events: list, buffer_anomaly_proc: list):
        super().__init__()

        self.buffer_traffic      = buffer_traffic
        self.buffer_events       = buffer_events
        self.buffer_anomaly_proc = buffer_anomaly_proc

        self.porog_traffic = 0.5
        self.porog_event   = 0.3

        self.presumably_traffic = []
        self.presumably_events  = []

        self.metrics = ["NoRATDetectRight", "NoRATDetectNoRight", "RATDetectRight", "RATDetectNoRight",
                        "NoRATReal", "RATReal"]
        self.real_rat_process = ["NingaliNETServer.exe", "RabbitHoleServer.exe", "RevengeRATServer.exe"]
        self.res_file = "GrafPrint\\resultsAnalyze.csv"

        while Path(self.res_file).exists():
            try:
                Path(self.res_file).unlink()
            except:
                continue

        col_name = f"index,{self.metrics[0]},{self.metrics[1]},{self.metrics[2]},{self.metrics[3]}," \
                   f"{self.real_rat_process[0]},{self.real_rat_process[1]}, {self.real_rat_process[2]},OtherProcess"
        with open(self.res_file, "w") as file:
            file.write(col_name)

    def ScannerBufferTraffic(self):
        if len(self.buffer_traffic) > 0:
            local_buffer = self.buffer_traffic.copy()
            self.buffer_traffic.clear()
            for i in range(len(local_buffer)):
                Time_Stamp_Start, Time_Stamp_End, Src_IP_Flow, Dst_IP_Flow, Src_Port_Flow, Dst_Port_Flow, loss = local_buffer[i]
                if loss >= self.porog_traffic:
                    self.presumably_traffic.append((Time_Stamp_Start, Time_Stamp_End, Src_IP_Flow,
                                                    Dst_IP_Flow, Src_Port_Flow, Dst_Port_Flow, True))
                else:
                    self.presumably_traffic.append((Time_Stamp_Start, Time_Stamp_End, Src_IP_Flow,
                                                    Dst_IP_Flow, Src_Port_Flow, Dst_Port_Flow, False))

    def ScannerBufferEvents(self):
        if len(self.buffer_events) > 0:
            local_buffer = self.buffer_events.copy()
            self.buffer_events.clear()
            for i in range(len(local_buffer)):
                Time_Stamp_Start, Time_Stamp_End, Process_Name, Direction_IP_Port, loss = local_buffer[i]
                arr_ip_port = []
                if isinstance(Direction_IP_Port, str):
                    if ";" in Direction_IP_Port:
                        Direction_IP_Port = f"{Direction_IP_Port}".split(";")
                    else:
                        Direction_IP_Port = [Direction_IP_Port]

                    for ip_port in Direction_IP_Port:
                        try:
                            ip, port = f"{ip_port}".split(":")
                            ip = int.from_bytes(IPv4Address(ip).packed, byteorder="big")
                            port = int(port)
                            arr_ip_port.append((ip, port))
                        except:
                            print(ip_port)

                if loss >= self.porog_event:
                    self.presumably_events.append((Time_Stamp_Start, Time_Stamp_End, Process_Name, arr_ip_port, True))
                else:
                    self.presumably_events.append((Time_Stamp_Start, Time_Stamp_End, Process_Name, arr_ip_port, False))

    def run(self):
        index = 0

        while True:
            interim_results = {"NoRATDetectRight": 0, "NoRATDetectNoRight": 0, "RATDetectRight": 0,
                               "RATDetectNoRight": 0, "NoRATReal": 0, "RATReal": 0,
                               "NingaliNETServer.exe": 0, "RabbitHoleServer.exe": 0, "RevengeRATServer.exe": 0
                               }

            anomaly_processes = []

            self.ScannerBufferTraffic()
            self.ScannerBufferEvents()

            if len(self.presumably_traffic) > 0 and len(self.presumably_events) > 0:
                count_events = len(self.presumably_events)

                for i in range(count_events):
                    Time_Start_Flow, Time_End_Flow, Process_Name, \
                        arr_ip_port, anomaly_proc = self.presumably_events[i]

                    if anomaly_proc:
                        Process = (Time_Start_Flow, Time_End_Flow, Process_Name, arr_ip_port)
                        anomaly_processes.append(Process)

                detection_RAT = False
                for j in range(len(self.presumably_traffic)):
                    Time_Start_Thread, Time_End_Thread, Src_IP_Flow, \
                        Dst_IP_Flow, Src_Port_Flow, Dst_Port_Flow, anomaly_traffic = self.presumably_traffic[j]

                    if anomaly_traffic:
                        for process in anomaly_processes:
                            Time_Start_Flow, Time_End_Flow, Process_Name, arr_ip_port = process
                            if len(arr_ip_port) > 0:
                                for ip_port in arr_ip_port:
                                    ip, port = ip_port

                                    if (Src_IP_Flow == ip or Dst_IP_Flow == ip) and \
                                            (Src_Port_Flow == port or Dst_Port_Flow == port):
                                        if not Process_Name in interim_results:
                                            interim_results[Process_Name] = 0
                                        interim_results[Process_Name] += 1
                                        detection_RAT = True

                            if not detection_RAT:
                                if Process_Name in self.real_rat_process:
                                    interim_results["NoRATDetectNoRight"] += 1
                                else:
                                    interim_results["NoRATDetectRight"] += 1

                            if Process_Name in self.real_rat_process:
                                interim_results["RATReal"] += 1
                            else:
                                interim_results["NoRATReal"] += 1

                self.buffer_anomaly_proc.append(interim_results)

                line = f"\n{index},{interim_results['NoRATDetectRight']},{interim_results['NoRATDetectNoRight']}," \
                       f"{interim_results['RATDetectRight']},{interim_results['RATDetectNoRight']}," \
                       f"{interim_results['NoRATReal']},{interim_results['RATReal']}," \
                       f"{interim_results['NingaliNETServer.exe']}," \
                       f"{interim_results['RabbitHoleServer.exe']}," \
                       f"{interim_results['RevengeRATServer.exe']},"
                OtherProcess = 0
                for anomaly in interim_results:
                    if anomaly == "browser.exe":
                        OtherProcess += interim_results[anomaly]
                line += str(OtherProcess)

                file = open(self.res_file, "a")
                file.write(line)
                file.close()

                index += 1

            time.sleep(5)