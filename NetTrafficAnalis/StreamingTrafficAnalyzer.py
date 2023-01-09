from ipaddress import IPv4Address
from NetTrafficAnalis.TrafficСharacts import CulcCharactsFlowOnWindow, \
    ParseTraffic, Packet_Charact, Flow_Charact, HUNDREDS_OF_NANOSECONDS
from threading import Thread
from pathlib import Path
from tqdm import tqdm

import subprocess as sp
import pandas as pd

import logging
import time
import re


class NetFlow:
    def __init__(self):
        """
            Возможные статусы потока:
            None - поток только создан, статус отсутствует
            0 - поток завершён с помощью флага fin
            -1 - поток завершён с помощью флага rst
            1 - поток в активном состоянии
            2 - поток прерван из-за временных рамок
        """
        self.current_flow = None
        self.status       = None
        self.pkts         = {}

    def append(self, pkt):
        if pkt[Packet_Charact.rst_flag] == 0 and pkt[Packet_Charact.fin_flag] == 0:
            self.pkts[self.current_flow].append(pkt)
        else:
            if pkt[Packet_Charact.rst_flag] == 1:
                self.status = -1
                self.pkts[self.current_flow].append(pkt)
            elif pkt[Packet_Charact.fin_flag] == 1:
                self.status = 0
                self.pkts[self.current_flow].append(pkt)


class NetFlows:
    def __init__(self, flow_time_limit):
        self.flows = {}
        self.flow_time_limit = flow_time_limit

    def creatFlow(self, pkt: dict, name_flow):
        if not name_flow in self.flows:
            self.flows[name_flow] = NetFlow()
        self.flows[name_flow].status = 1
        self.flows[name_flow].current_flow = pkt[Packet_Charact.timestamp]
        self.flows[name_flow].pkts[self.flows[name_flow].current_flow] = list()
        self.flows[name_flow].append(pkt)

    def delFlow(self, name_flow, timeFlow):
        del self.flows[name_flow].pkts[timeFlow]
        if not self.flows[name_flow].pkts:
            del self.flows[name_flow]

    def appendPacket(self, pkt: dict):
        name_flow = pkt[Packet_Charact.ip_src] + pkt[Packet_Charact.ip_dst] + pkt[Packet_Charact.transp_protocol] + \
                    pkt[Packet_Charact.port_src] + pkt[Packet_Charact.port_dst]

        if not name_flow in self.flows:
            self.creatFlow(pkt, name_flow)
        else:
            if self.flows[name_flow].status == 1:
                if pkt[Packet_Charact.timestamp] - self.flows[name_flow].current_flow >= self.flow_time_limit:
                    self.flows[name_flow].status = 2
                    self.creatFlow(pkt, name_flow)
                else:
                    self.flows[name_flow].append(pkt)
            else:
                self.creatFlow(pkt, name_flow)

    def printInFile(self, file_name):
        max_subflow = 0
        add_pkts = 0
        with open(file_name, "w") as printf:
            for nameFlow in self.flows:
                index = 0
                printf.write(f"{nameFlow} = {'{'}\n")
                for timeFlow in self.flows[nameFlow].pkts:
                    index += 1
                    if index > max_subflow:
                        max_subflow = index
                    printf.write(f"\t{timeFlow}: {'{'}\n")
                    for idx in range(len(self.flows[nameFlow].pkts[timeFlow])):
                        pkt = self.flows[nameFlow].pkts[timeFlow][idx]
                        add_pkts += 1
                        printf.write(f"\t\t{idx}: {'{'}\n")
                        for ch in Packet_Charact:
                            printf.write(f"\t\t\t{ch}: {pkt[ch]},\n")
                        printf.write(f"\t\t{'},'}\n")
                    printf.write(f"\t{'},'}\n")
                printf.write(f"{'}'}\n")

        print(f"Максимальное количество потоков с одинаковым именем: {max_subflow}")
        print(f"Было записано {add_pkts} пакетов")
        print("Запись завершена!")


class SnifferTraffic(Thread):
    def __init__(self, pcap_length, iface_name, path_tshark, trffic_file_mask, path_name):
        super().__init__()

        self.pcap_length       = pcap_length
        self.iface_name        = iface_name
        self.trffic_file_mask  = trffic_file_mask
        self.path_name         = path_name
        self.path_tshark       = path_tshark

        self.last_file_id   = None
        self.iface          = None
        self.th_main_sniff  = None
        self.run_sniff      = False

    def __SniffPackets__(self, file_name):
        sniffer = [self.path_tshark, "-c", str(self.pcap_length), "-w", file_name]
        CREATE_NO_WINDOW = 0x08000000
        if self.iface:
            sniffer.append("-i " + str(self.iface))
        prog = sp.Popen(sniffer, creationflags=CREATE_NO_WINDOW)
        prog.communicate()

    def GetNumberIface(self):
        get_ifaces = [self.path_tshark, "-D"]
        CREATE_NO_WINDOW = 0x08000000
        prog = sp.Popen(get_ifaces, stdout=sp.PIPE, creationflags=CREATE_NO_WINDOW)
        ifaces, err = prog.communicate()
        for iface in ifaces.decode("utf_8").split("\n"):
            if "(" + self.iface_name + ")" in iface:
                number_iface = iface[0]
                self.iface = number_iface

    def GetLastFileId(self):
        """
            Функция получения индекса последнего pcapng файла в директории сборщика,
            оставшихся после предыдущего этапа работы программы (в случае если программа не успела
            завершить обработку всех pcapng файлов и была завершена).
            Также данная функция составляет очередь preprocessing_queue для возобновления ранее
            прерванного процесса обработки файлов трафика.
            Принимает:
                sniffer_home - путь к директории сборщика;
            Возвращает
                indexs_files_pcapng[-1] - индекс последнего pcapng файла в дирректории.
                Если таковых файлов нет, то возвращает 0
        """

        path_sniffer_home = Path(self.path_name)
        file_arr = []

        for file in path_sniffer_home.iterdir():
            file = str(file)
            if "tmp" in file:
                continue
            elif not (self.trffic_file_mask in file):
                continue
            else:
                file_arr.append(file)

        if len(file_arr) > 0:
            indexs_files_pcapng = []

            # Получение индексов файлов с трафиком
            for file_name in file_arr:
                index = file_name.find(self.trffic_file_mask)
                index_file = [int(s) for s in re.split('_|.p', file_name[index:]) if s.isdigit()][0]
                indexs_files_pcapng.append(index_file)

            indexs_files_pcapng.sort()

            self.last_file_id = indexs_files_pcapng[-1]
        else:
            self.last_file_id = -1

    def run(self):
        print("Поток сбора трафика запущен")
        if Path(self.path_name).exists():
            if not Path(self.path_name+"\\tmp").exists():
                Path(self.path_name + "\\tmp").mkdir()

            self.GetNumberIface()
            self.GetLastFileId()

            self.last_file_id += 1
            while True:
                traffic_file = f"{self.path_name}\\{self.trffic_file_mask}{self.last_file_id}.pcapng"
                traffic_file_tmp = f"{self.path_name}\\tmp\\{self.trffic_file_mask}{self.last_file_id}.pcapng"

                try:
                    self.__SniffPackets__(traffic_file_tmp)
                    Path(traffic_file_tmp).rename(traffic_file)
                    self.last_file_id += 1
                except Exception as err:
                    print("Ошибка во время сбора трафика! %s" % str(err))
                    continue

        else:
            print("Директория для сбора трафика не существует, создайте её и перезапустите процесс!")


class AnalyzerPackets(Thread):
    def __init__(self, flow_time_limit, charact_file_length,
                 traffic_waiting_time, charact_file_mask, ip_client, path_name):
        super().__init__()

        self.flow_time_limit        = flow_time_limit
        self.charact_file_length    = charact_file_length
        self.charact_file_mask      = charact_file_mask
        self.path_name              = path_name

        for idx in range(len(ip_client)):
            if not isinstance(ip_client[idx], IPv4Address):
                ip_client[idx] = IPv4Address(ip_client[idx])

        self.ip_client = [int.from_bytes(ip.packed, byteorder="big") for ip in ip_client]

        self.files_traffic_arr  = list()
        self.array_paket_global = list()
        self.NetFlows_obj       = NetFlows(self.flow_time_limit)

        self.index_charact_file   = 0
        self.traffic_waiting_time = traffic_waiting_time

        self.GetLastFileId()
        self.GetFilesTraffic()

    def GetFilesTraffic(self):
        path_sniffer_home = Path(self.path_name)
        files_local      = {}
        files_timecreate = []

        for file in path_sniffer_home.iterdir():
            file_split = str(file).split(".")

            if file.is_file():
                if file_split[1] == "pcap" or file_split[1] == "pcapng":
                    time_create = file.stat().st_mtime
                    files_timecreate.append(time_create)
                    files_local[time_create] = str(file)
                else:
                    continue
            else:
                continue

        self.files_traffic_arr = []
        if len(files_timecreate) > 0:
            files_timecreate.sort()

            for tc in files_timecreate:
                self.files_traffic_arr.append(files_local[tc])

        return len(self.files_traffic_arr)

    def GetLastFileId(self):
        path_home = Path(self.path_name)
        file_arr = []

        for file in path_home.iterdir():
            file = str(file)
            if not (".csv" in file):
                continue
            else:
                file_arr.append(file)

        if len(file_arr) > 0:
            indexs_files_charact = []

            # Получение индексов файлов с трафиком
            for file_name in file_arr:
                index_file = int(file_name.split(".")[0].split("\\")[-1].split("_")[-1])
                indexs_files_charact.append(index_file)

            indexs_files_charact.sort()

            self.index_charact_file = indexs_files_charact[-1]
        else:
            self.index_charact_file = -1

    def ParseTrafficFile(self, file_name):
        print(f"Загружается трафик: {file_name}")
        bar = tqdm(desc="Процесс загрузки")
        for paket in ParseTraffic(file_name):
            self.array_paket_global.append(paket)
            bar.update(1)
        bar.close()

        # path_new = self.path_name + "\\" + "Обработанные файлы"
        # if not Path(path_new).exists():
        #     Path(path_new).mkdir()
        # file_only_name = file_name.split("\\")[-1]
        # Path(file_name).rename(path_new + "\\" + file_only_name)

    def ProcessingTraffic(self, arr_traffic_file=None):
        if not arr_traffic_file is None:
            for file in arr_traffic_file:
                self.ParseTrafficFile(file)

        for pkt in self.array_paket_global:
            self.NetFlows_obj.appendPacket(pkt)
        self.array_paket_global.clear()
        # NetFlows_obj.printInFile("print.py")

        bar = tqdm(desc="Процесс обработки")
        array_characts = list()
        for nameFlow in list(self.NetFlows_obj.flows):
            if not nameFlow in self.NetFlows_obj.flows:
                continue

            for timeFlow in list(self.NetFlows_obj.flows[nameFlow].pkts):
                if not timeFlow in self.NetFlows_obj.flows[nameFlow].pkts:
                    continue

                # if self.NetFlows_obj.flows[nameFlow].current_flow == timeFlow and \
                #         self.NetFlows_obj.flows[nameFlow].status == 1:
                #     continue
                # else:
                flow = self.NetFlows_obj.flows[nameFlow].pkts[timeFlow]
                flowCharacts = CulcCharactsFlowOnWindow(flow, self.ip_client)
                self.NetFlows_obj.delFlow(nameFlow, timeFlow)
                array_characts.append(flowCharacts)
                bar.update(1)
        bar.close()

        print(f"Выявлено характеристик: {len(array_characts)}")

        if len(array_characts) == 0:
            print("Не выделено ни одного набора характеристик!")
            return False
        else:
            try:
                self.index_charact_file += 1
                characts_file_name = self.path_name + "\\" + \
                                     self.charact_file_mask + str(self.index_charact_file) + ".csv"

                pd_characts = pd.DataFrame(array_characts)
                pd_characts = pd_characts.sort_values(by=Flow_Charact.Time_Stamp_End)
                pd_characts.to_csv(characts_file_name, index=False)

                print("Парсинг завершился!")

                return True
            except Exception as err:
                logging.exception(f"Ошибка!\n{err}")
                return False

    def PaketsAnalyz(self, pakets):
        for pkt in pakets:
            self.array_paket_global.append(pkt)

        if Path(self.path_name).exists():
            print("Запущен анализ заданных пакетов")
            self.ProcessingTraffic()
        else:
            print("Директория с трафиком для анализа не существует, видимо процесс сбора трафика не был запущен ранее!")

    def run(self):
        print("Поток анализа трафика запущен")
        if Path(self.path_name).exists():
            while True:
                count_file_traffic = self.GetFilesTraffic()
                if count_file_traffic == 0:
                    time.sleep(self.traffic_waiting_time)
                    continue
                else:
                    try:
                        self.ProcessingTraffic(self.files_traffic_arr)
                        self.files_traffic_arr.clear()
                    except IndexError:
                        continue
                return
        else:
            print("Директория с трафиком для анализа не существует, видимо процесс сбора трафика не был запущен ранее!")


if __name__ == '__main__':
    path_name = "F:\\TRAFFIC\\VNAT\\"

    # Параметры сборщика трафика
    # size_pcap_length  = 10000
    # iface_name        = "VMware_Network_Adapter_VMnet3"
    # trffic_file_mask  = "traffic_"
    # path_tshark       = "Wireshark\\tshark.exe"
    #
    # sniffer = SnifferTraffic(size_pcap_length, iface_name, path_tshark, trffic_file_mask, path_name)
    # sniffer.start()

    # Параметры анализатора трафика
    flow_time_limit         = 1 * 60 * HUNDREDS_OF_NANOSECONDS
    traffic_waiting_time    = 200
    charact_file_length     = 10000000000

    # charact_file_name       = "RAT_revenge_"
    # ip_client               = [IPv4Address("192.168.10.128")]

    # charact_file_name     = "nonvpn_rsync"
    # VNAT клиенты
    ip_client = [IPv4Address("10.101.1.100"), IPv4Address("10.103.1.2"), IPv4Address("10.102.1.2"),
                 IPv4Address("10.104.1.158"), IPv4Address("10.104.1.2"), IPv4Address("10.105.1.2"),
                 IPv4Address("10.113.1.150"), IPv4Address("10.115.1.2"), IPv4Address("10.114.1.1"),
                 IPv4Address("10.115.1.123"), IPv4Address("10.118.1.100"), IPv4Address("10.118.1.2"),
                 IPv4Address("10.116.1.2"), IPv4Address("10.117.1.1"), IPv4Address("10.121.1.130"),
                 IPv4Address("10.121.1.145"), IPv4Address("10.119.1.197"), IPv4Address("10.119.1.2"),
                 IPv4Address("10.120.1.1"), IPv4Address("10.122.1.103"), IPv4Address("10.122.1.2"),
                 IPv4Address("10.123.1.2"), IPv4Address("192.168.0.180")]

    # ip_client             = [IPv4Address("192.168.20.131"), IPv4Address("192.168.20.132"),
    #                          IPv4Address("192.168.20.133"), IPv4Address("192.168.20.134")]

    for charact_file_name in Path(path_name).iterdir():
        analizator = AnalyzerPackets(flow_time_limit, charact_file_length, traffic_waiting_time,
                                     str(charact_file_name).split("\\")[-1], ip_client, str(charact_file_name))
        analizator.run()





