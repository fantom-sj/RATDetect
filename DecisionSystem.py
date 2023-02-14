import logging
from ipaddress import IPv4Address
from threading import Thread
import math

import time


class DecisionSystem(Thread):
    def __init__(self, ip_client: list, buffer_traffic: list, buffer_events: list, buffer_output: dict):
        super().__init__()

        self.run_toggle = False

        self.ip_client_number = [int.from_bytes(IPv4Address(ip).packed, byteorder='big') for ip in ip_client]

        self.buffer_traffic      = buffer_traffic
        self.buffer_events       = buffer_events

        self.porog_traffic = 0.7
        self.porog_event   = 0.2

        self.buffer_output               = buffer_output
        self.buffer_output["traffic"]    = list()
        self.buffer_output["events"]     = list()

        self.ID_events  = 0
        self.ID_traffic = 0

        self.real_rat_process = ["NingaliNETServer.exe", "RabbitHoleServer.exe", "RevengeRATServer.exe"]

        self.buffer_time = {
            "Process": {
                "RevengeRATServer.exe": list()
            },

            "NetFlows": {
                frozenset({int.from_bytes(IPv4Address("192.168.10.128").packed, byteorder="big"),
                           int.from_bytes(IPv4Address("192.168.10.129").packed, byteorder="big"), 333}): list()
            },
        }

        self.buffer_output["AnalysisResults"] = {
            "Process": {
                "RevengeRATServer.exe": {
                    "anomal_active": 0,
                    "normal_active": 0,
                    "netflows":      list()
                }
            },



            "NetFlows": {
                frozenset({int.from_bytes(IPv4Address("192.168.10.128").packed, byteorder="big"),
                           int.from_bytes(IPv4Address("192.168.10.129").packed, byteorder="big"), 333}): {
                    "anomal_active": 0,
                    "normal_active": 0
                }
            },

            "RATtrojans": {
                frozenset({"RevengeRATServer.exe", "192.168.10.128"}): {
                    "PercentAnomalyEvents":  0.0,
                    "PercentAnomalyNetFlow": 0.0,
                    "PossibilityRAT":        0.0
                }
            }
        }

        self.buffer_output["statistics"] = {
            "CorrectDetectSafeProcesses":   0,
            "NoCorrectDetectSafeProcesses": 0,
            "CorrectDetectRAT_Trojans":     0,
            "NoCorrectDetectRAT_Trojans":   0,
        }

    def ScannerBufferTraffic(self):
        if len(self.buffer_traffic) > 0:
            local_buffer = self.buffer_traffic.copy()
            self.buffer_traffic.clear()
            for i in range(len(local_buffer)):
                Time_Stamp_Start, Time_Stamp_End, Src_IP_Flow, Dst_IP_Flow, \
                    Src_Port_Flow, Dst_Port_Flow, loss = local_buffer[i]

                if Src_IP_Flow in self.ip_client_number:
                    port = Dst_Port_Flow
                elif Dst_IP_Flow in self.ip_client_number:
                    port = Src_Port_Flow
                else:
                    continue

                name_netflow = frozenset({Src_IP_Flow, Dst_IP_Flow, port})
                if loss >= self.porog_traffic:
                    anomaly_netflow = 1
                else:
                    anomaly_netflow = 0

                data_netflow = {"ID":               self.ID_traffic,
                                "Time_Stamp_Start": Time_Stamp_Start,
                                "Time_Stamp_End":   Time_Stamp_End,
                                "Src_IP_Flow":      Src_IP_Flow,
                                "Dst_IP_Flow":      Dst_IP_Flow,
                                "Src_Port_Flow":    Src_Port_Flow,
                                "Dst_Port_Flow":    Dst_Port_Flow,
                                "loss":             float(loss),
                                "anomaly":          anomaly_netflow}

                self.buffer_output["traffic"].append(data_netflow)

                if not (name_netflow in self.buffer_output["AnalysisResults"]["NetFlows"]):
                    self.buffer_output["AnalysisResults"]["NetFlows"][name_netflow]                  = dict()
                    self.buffer_output["AnalysisResults"]["NetFlows"][name_netflow]["anomal_active"] = 0
                    self.buffer_output["AnalysisResults"]["NetFlows"][name_netflow]["normal_active"] = 0
                    self.buffer_time["NetFlows"][name_netflow]                                       = list()

                if anomaly_netflow:
                    self.buffer_output["AnalysisResults"]["NetFlows"][name_netflow]["anomal_active"] += 1
                else:
                    self.buffer_output["AnalysisResults"]["NetFlows"][name_netflow]["normal_active"] += 1

                time_ = math.trunc(Time_Stamp_Start / (10**9))
                self.buffer_time["NetFlows"][name_netflow].append(time_)

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
                            host_ip, ip, port = f"{ip_port}".split(":")
                            host_ip           = int.from_bytes(IPv4Address(host_ip).packed, byteorder="big")
                            ip                = int.from_bytes(IPv4Address(ip).packed, byteorder="big")
                            port              = int(port)
                            arr_ip_port.append((host_ip, ip, port))
                        except Exception as err:
                            logging.exception(err)
                            print(ip_port)

                if len(arr_ip_port) == 0:
                    connection = 0
                else:
                    connection = arr_ip_port

                if loss >= self.porog_event:
                    anomal_process = 1
                else:
                    anomal_process = 0

                data_process = {"ID":               self.ID_events,
                                "Time_Stamp_Start": Time_Stamp_Start,
                                "Time_Stamp_End":   Time_Stamp_End,
                                "Process_Name":     Process_Name,
                                "connection":       connection,
                                "loss":             float(loss),
                                "anomaly":          anomal_process}

                self.buffer_output["events"].append(data_process)

                if not (Process_Name in self.buffer_output["AnalysisResults"]["Process"]):
                    self.buffer_output["AnalysisResults"]["Process"][Process_Name]                  = dict()
                    self.buffer_output["AnalysisResults"]["Process"][Process_Name]["anomal_active"] = 0
                    self.buffer_output["AnalysisResults"]["Process"][Process_Name]["normal_active"] = 0
                    self.buffer_output["AnalysisResults"]["Process"][Process_Name]["netflows"]      = list()
                    self.buffer_time["Process"][Process_Name]                                       = list()

                for ip_port in arr_ip_port:
                    host_ip, ip, port = ip_port
                    self.buffer_output["AnalysisResults"]["Process"][Process_Name]["netflows"].append((host_ip,
                                                                                                       ip, port))

                if anomal_process:
                    self.buffer_output["AnalysisResults"]["Process"][Process_Name]["anomal_active"] += 1
                else:
                    self.buffer_output["AnalysisResults"]["Process"][Process_Name]["normal_active"] += 1

                time_ = math.trunc(Time_Stamp_Start / (10**9))
                self.buffer_time["Process"][Process_Name].append(time_)

                self.ID_events += 1

    def run(self):
        print("Подсистема принятия решения запущена!")

        while self.run_toggle:
            self.ScannerBufferTraffic()
            self.ScannerBufferEvents()

            for proc_name in self.buffer_output["AnalysisResults"]["Process"]:
                for flow in self.buffer_output["AnalysisResults"]["Process"][proc_name]["netflows"]:
                    host_ip, ip, port = flow
                    name_flow = frozenset({host_ip, ip, port})

                    times_proc = self.buffer_time["Process"][proc_name]

                    if name_flow in self.buffer_output["AnalysisResults"]["NetFlows"]:
                        if (self.buffer_output["AnalysisResults"]["Process"][proc_name]["anomal_active"] > 0) and \
                           (self.buffer_output["AnalysisResults"]["NetFlows"][name_flow]["anomal_active"] > 0):
                            anomal_events = self.buffer_output["AnalysisResults"]["Process"][proc_name]["anomal_active"]
                            normal_events = self.buffer_output["AnalysisResults"]["Process"][proc_name]["normal_active"]
                            if anomal_events != 0 and normal_events != 0:
                                AnomalyEvents = anomal_events / (anomal_events + normal_events)
                            elif anomal_events != 0 and normal_events == 0:
                                AnomalyEvents = 1
                            elif anomal_events == 0 and normal_events != 0:
                                AnomalyEvents = 0
                            else:
                                AnomalyEvents = -1

                            anomal_flow = self.buffer_output["AnalysisResults"]["NetFlows"][name_flow]["anomal_active"]
                            normal_flow = self.buffer_output["AnalysisResults"]["NetFlows"][name_flow]["normal_active"]
                            if anomal_flow != 0 and normal_flow != 0:
                                AnomalyNetFlow = anomal_flow / (anomal_flow + normal_flow)
                            elif anomal_flow != 0 and normal_flow == 0:
                                AnomalyNetFlow = 1
                            elif anomal_flow == 0 and normal_flow != 0:
                                AnomalyNetFlow = 0
                            else:
                                AnomalyNetFlow = -1

                            times_flow = self.buffer_time["NetFlows"][name_flow]
                            sovpaden = 0


                            if AnomalyEvents > 0 and AnomalyNetFlow > 0:
                                # PossibilityRAT = AnomalyEvents * AnomalyNetFlow

                                for time_proc in times_proc:
                                    if time_proc in times_flow:
                                        sovpaden += 1

                                if len(times_proc) == 0:
                                    PossibilityRAT = 0
                                else:
                                    PossibilityRAT = sovpaden / len(times_proc)
                            else:
                                PossibilityRAT = 0

                            if PossibilityRAT > 0:
                                rat_name = frozenset({proc_name, str(IPv4Address(host_ip))})
                                if not (rat_name in self.buffer_output["AnalysisResults"]["RATtrojans"]):
                                    self.buffer_output["AnalysisResults"]["RATtrojans"][rat_name] = dict()

                                self.buffer_output["AnalysisResults"]["RATtrojans"][rat_name]\
                                    ["PercentAnomalyEvents"] = AnomalyEvents * 100
                                self.buffer_output["AnalysisResults"]["RATtrojans"][rat_name]\
                                    ["PercentAnomalyNetFlow"] = AnomalyNetFlow * 100
                                self.buffer_output["AnalysisResults"]["RATtrojans"][rat_name]\
                                    ["PossibilityRAT"] = PossibilityRAT

                            if PossibilityRAT > 0.5:
                                if proc_name in self.real_rat_process:
                                    self.buffer_output["statistics"]["CorrectDetectRAT_Trojans"]     += 1
                                else:
                                    self.buffer_output["statistics"]["NoCorrectDetectSafeProcesses"] += 1
                            else:
                                if proc_name in self.real_rat_process:
                                    self.buffer_output["statistics"]["NoCorrectDetectRAT_Trojans"]   += 1
                                else:
                                    self.buffer_output["statistics"]["CorrectDetectSafeProcesses"]   += 1

            time.sleep(2)

        print("Подсистема принятия решения остановлена!")