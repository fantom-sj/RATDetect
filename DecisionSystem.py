import logging
import time
from threading import Thread
from pathlib import Path
import os

class DecisionSystem(Thread):
    def __init__(self, buffer_traffic: list, buffer_events: list, buffer_anomaly_proc: list):
        super().__init__()

        self.buffer_traffic      = buffer_traffic
        self.buffer_events       = buffer_events
        self.buffer_anomaly_proc = buffer_anomaly_proc

        self.porog_traffic = 0.01
        self.porog_event   = 0.0888

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
                traffic_res = local_buffer[i]
                for time_start in traffic_res:
                    loss, time_end, direction_ip_port = traffic_res[time_start]
                    if loss >= self.porog_traffic:
                        self.presumably_traffic.append((time_start, time_end, direction_ip_port, True))
                    else:
                        self.presumably_traffic.append((time_start, time_end, direction_ip_port, False))

    def ScannerBufferEvents(self):
        if len(self.buffer_events) > 0:
            local_buffer = self.buffer_events.copy()
            self.buffer_events.clear()
            for i in range(len(local_buffer)):
                event_res = local_buffer[i]
                for proc_name in event_res:
                    loss, time_stars, time_end, direction_ip_port = event_res[proc_name]
                    if loss >= self.porog_event:
                        self.presumably_events.append((proc_name, time_stars, time_end, direction_ip_port, True))
                    else:
                        self.presumably_events.append((proc_name, time_stars, time_end, direction_ip_port, False))

    def run(self):
        index = 0

        while True:
            interim_results = {"NoRATDetectRight": 0, "NoRATDetectNoRight": 0, "RATDetectRight": 0,
                               "RATDetectNoRight": 0, "NoRATReal": 0, "RATReal": 0,
                               "NingaliNETServer.exe": 0, "RabbitHoleServer.exe": 0, "RevengeRATServer.exe": 0}

            self.ScannerBufferTraffic()
            self.ScannerBufferEvents()
            if len(self.presumably_traffic) > 0 and len(self.presumably_events) > 0:

                count_events = len(self.presumably_events)
                for i in range(count_events):
                    proc_name, event_stars, event_end, event_direction, anomaly_proc = self.presumably_events[i]
                    detection_RAT = False
                    for j in range(len(self.presumably_traffic)):
                        traffic_start, traffic_end, traffic_direction, anomaly_traffic = self.presumably_traffic[j]
                        if anomaly_proc and anomaly_traffic:
                            if isinstance(event_direction, list) and isinstance(traffic_direction, list):
                                if traffic_start <= event_stars < event_end <= traffic_end:
                                    for event_dir in event_direction:
                                        if event_dir in traffic_direction:
                                            if not proc_name in interim_results:
                                                interim_results[proc_name] = 0
                                            interim_results[proc_name] += 1
                                            if proc_name in self.real_rat_process:
                                                interim_results["RATDetectRight"] += 1
                                            else:
                                                interim_results["RATDetectNoRight"] += 1
                                            detection_RAT = True
                                    break

                    if not detection_RAT:
                        if proc_name in self.real_rat_process:
                            interim_results["NoRATDetectNoRight"] += 1
                        else:
                            interim_results["NoRATDetectRight"] += 1

                    if proc_name in self.real_rat_process:
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
            time.sleep(20)