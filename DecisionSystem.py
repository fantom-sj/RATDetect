import logging
import time
from threading import Thread
from pathlib import Path
from ipaddress import IPv4Address
import os
import numpy as np

class DecisionSystem(Thread):
    def __init__(self, buffer_traffic: list, buffer_events: list, buffer_output: dict):
        super().__init__()

        self.buffer_traffic      = buffer_traffic
        self.buffer_events       = buffer_events

        self.porog_traffic = 0.2
        self.porog_event   = 0.3

        self.presumably_traffic = list()
        self.presumably_events  = list()

        self.buffer_output = buffer_output
        self.buffer_output["traffic"]    = list()
        self.buffer_output["events"]     = list()
        self.buffer_output["statistics"] = dict()

        self.ID_events  = 0
        self.ID_traffic = 0

        self.metrics = ["NoRATDetectRight", "NoRATDetectNoRight", "RATDetectRight", "RATDetectNoRight",
                        "NoRATReal", "RATReal"]
        self.real_rat_process = ["NingaliNETServer.exe", "RabbitHoleServer.exe", "RevengeRATServer.exe"]

    def ScannerBufferTraffic(self):
        if len(self.buffer_traffic) > 0:
            local_buffer = self.buffer_traffic.copy()
            self.buffer_traffic.clear()
            for i in range(len(local_buffer)):
                Time_Stamp_Start, Time_Stamp_End, Src_IP_Flow, Dst_IP_Flow, Src_Port_Flow, Dst_Port_Flow, loss = local_buffer[i]
                if loss >= self.porog_traffic:
                    self.presumably_traffic.append((Time_Stamp_Start, Time_Stamp_End, Src_IP_Flow,
                                                    Dst_IP_Flow, Src_Port_Flow, Dst_Port_Flow, True))
                    self.buffer_output["traffic"].append({"ID":                 self.ID_traffic,
                                                          "Time_Stamp_Start":   Time_Stamp_Start,
                                                          "Time_Stamp_End":     Time_Stamp_End,
                                                          "Src_IP_Flow":        Src_IP_Flow,
                                                          "Dst_IP_Flow":        Dst_IP_Flow,
                                                          "Src_Port_Flow":      Src_Port_Flow,
                                                          "Dst_Port_Flow":      Dst_Port_Flow,
                                                          "loss": float(loss),
                                                          "anomaly":            1})
                else:
                    self.presumably_traffic.append((Time_Stamp_Start, Time_Stamp_End, Src_IP_Flow,
                                                    Dst_IP_Flow, Src_Port_Flow, Dst_Port_Flow, False))
                    self.buffer_output["traffic"].append({"ID":                 self.ID_traffic,
                                                          "Time_Stamp_Start":   Time_Stamp_Start,
                                                          "Time_Stamp_End":     Time_Stamp_End,
                                                          "Src_IP_Flow":        Src_IP_Flow,
                                                          "Dst_IP_Flow":        Dst_IP_Flow,
                                                          "Src_Port_Flow":      Src_Port_Flow,
                                                          "Dst_Port_Flow":      Dst_Port_Flow,
                                                          "loss":               float(loss),
                                                          "anomaly":            0})
                self.ID_traffic += 1

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

                if len(arr_ip_port) == 0:
                    connection = 0
                else:
                    connection = arr_ip_port

                if loss >= self.porog_event:
                    self.presumably_events.append((Time_Stamp_Start, Time_Stamp_End, Process_Name, arr_ip_port, True))
                    self.buffer_output["events"].append({"ID":               self.ID_events,
                                                         "Time_Stamp_Start": Time_Stamp_Start,
                                                         "Time_Stamp_End":   Time_Stamp_End,
                                                         "Process_Name":     Process_Name,
                                                         "connection":       connection,
                                                         "loss":             float(loss),
                                                         "anomaly":          1})
                else:
                    self.presumably_events.append((Time_Stamp_Start, Time_Stamp_End, Process_Name, arr_ip_port, False))
                    self.buffer_output["events"].append({"ID":               self.ID_events,
                                                         "Time_Stamp_Start": Time_Stamp_Start,
                                                         "Time_Stamp_End":   Time_Stamp_End,
                                                         "Process_Name":     Process_Name,
                                                         "connection":       connection,
                                                         "loss":             float(loss),
                                                         "anomaly":          0})
                self.ID_events += 1

    def run(self):
        results = {
            "SafeProcesses":        [],
            "SuspiciousProcesses":  [],
            "SafeNetFlow":          [],
            "SuspiciousNetFlow":    [],

            "RAT_trojans": {"NingaliNETServer.exe": 0, "RabbitHoleServer.exe": 0, "RevengeRATServer.exe": 0},

            "CorrectDetectSafeProcesses":   0,
            "NoCorrectDetectSafeProcesses":   0,
            "CorrectDetectRAT_Trojans":   0,
            "NoCorrectDetectRAT_Trojans":   0,
        }

        while True:
            self.ScannerBufferTraffic()
            self.ScannerBufferEvents()

            for event in self.presumably_events:
                _, _, _, _, anomal = event
                if anomal and not (event in results["SuspiciousProcesses"]):
                    results["SuspiciousProcesses"].append(event)
                elif not anomal and not (event in results["SafeProcesses"]):
                    results["SafeProcesses"].append(event)

            for netflow in self.presumably_traffic:
                _, _, _, _, _, _, anomal = netflow
                if anomal and not (netflow in results["SuspiciousNetFlow"]):
                    results["SuspiciousNetFlow"].append(netflow)
                elif not anomal and not (netflow in results["SafeNetFlow"]):
                    results["SafeNetFlow"].append(netflow)

            for event in results["SuspiciousProcesses"]:
                _, _, Process_Name, arr_ip_port, _ = event
                for ip_port in arr_ip_port:
                    ip, port = ip_port

                    # if Process_Name == "RevengeRATServer.exe":
                    #     print(f"{Process_Name}: {str(IPv4Address(ip))}, {port}")

                    for netflow in results["SuspiciousNetFlow"]:
                        Time_Stamp_Start, Time_Stamp_End, Src_IP_Flow, \
                            Dst_IP_Flow, Src_Port_Flow, Dst_Port_Flow, _ = netflow

                        if (ip == Src_IP_Flow or ip == Dst_IP_Flow) and \
                           (port == Src_Port_Flow or port == Dst_Port_Flow):
                            if not Process_Name in results["RAT_trojans"]:
                                results["RAT_trojans"][Process_Name] = 0
                            results["RAT_trojans"][Process_Name] += 1

            self.buffer_output["statistics"].clear()
            for record in results:
                self.buffer_output["statistics"][record] = results[record]

            time.sleep(1)