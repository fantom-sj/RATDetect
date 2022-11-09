from ipaddress import IPv4Address, ip_address, IPv6Address
from SnifferPaket.characts import CulcCharactsOnWindow, CHARACTERISTIC
from threading import Thread
from pathlib import Path

import subprocess as sp
import pandas as pd

import logging
import time
import dpkt
import math
import re


class Sniffer:
    def __init__(self, size_pcap_length, iface_name, trffic_name, trffic_file_mask, path_name):
        self.size_pcap_length           = size_pcap_length
        self.iface_name                 = iface_name
        self.trffic_name                = trffic_name
        self.trffic_file_mask           = trffic_file_mask
        self.path_name                  = path_name

        self.last_file_id   = None
        self.iface          = None
        self.th_main_sniff  = None
        self.run_sniff      = False

    def __SniffPackets__(self, file_name):
        sniffer = ["Wireshark\\tshark.exe", "-c", str(self.size_pcap_length), "-w", file_name]
        CREATE_NO_WINDOW = 0x08000000
        if self.iface:
            sniffer.append("-i " + str(self.iface))
        prog = sp.Popen(sniffer, creationflags=CREATE_NO_WINDOW)
        prog.communicate()

    def GetNumberIface(self):
        get_ifaces = ["Wireshark\\tshark.exe", "-D"]
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

        path_sniffer_home = Path(self.path_name + "\\" + self.trffic_name)
        file_arr = []

        for file in path_sniffer_home.iterdir():
            file = str(file)
            if not (self.trffic_file_mask in file):
                continue
            else:
                file_arr.append(file)

        if len(file_arr) > 0:
            indexs_files_pcapng = []

            # Получение индексов файлов с трафиком
            for file_name in file_arr:
                index_SnHome = file_name.find(self.trffic_name)
                index_file = [int(s) for s in re.split('_|.p', file_name[index_SnHome + 5:]) if s.isdigit()][0]
                indexs_files_pcapng.append(index_file)

            indexs_files_pcapng.sort()

            # for index in indexs_files_pcapng:
            #     traffic_file = f"{self.path_name}\\{self.trffic_name}\\{self.trffic_file_mask}{index}.pcapng"
            #     preprocessing_queue.append(traffic_file)

            self.last_file_id = indexs_files_pcapng[-1]
        else:
            self.last_file_id = -1

    def SniffLoop(self):
        """
            Функция, которая сама запускается в отдельном потоке для контроля за сбором трафика,
            процесс запуска которого также помещается в отдельный поток. По окончанию сбора пакетов
            происходит Запись имени файла в очередь на предварительную обработку, которая осуществляется
            в основном потоке программы.

            Принимает:
                sniffer_home - путь к директории сборщика;
                index_last_file - индекс последнего файла с трафиком, который хранится в директории сборщика;
                num_iface - индекс интерфейса для сбора трафика;
                count_pkt_one_pcap - количество пакетов сохраняемых в 1 файл;
            Ничего не возвращает, работает в бесконечном цикле внутри своего потока.
        """

        self.last_file_id += 1
        while self.run_sniff:
            traffic_file = f"{self.path_name}\\{self.trffic_name}\\{self.trffic_file_mask}{self.last_file_id}.pcapng"

            try:
                th_sniff = Thread(target=self.__SniffPackets__, args=(traffic_file,))
                th_sniff.start()
                th_sniff.join()
                # preprocessing_queue.append(traffic_file)

                self.last_file_id += 1
            except Exception as err:
                print("Ошибка во время снифинга! %s" % str(err))
                continue

    def run(self):
        if Path(self.path_name).exists():
            if not Path(self.path_name + "\\" + self.trffic_name).exists():
                Path(self.path_name + "\\" + self.trffic_name).mkdir()

            self.GetNumberIface()
            print(f"Интерфейс {self.iface_name} имеет ID: {self.iface}")
            self.GetLastFileId()
            print(f"Индекс последнего файла с траффиком: {self.last_file_id}")

            self.run_sniff = True
            self.th_main_sniff = Thread(target=self.SniffLoop, args=())
            self.th_main_sniff.start()
            print("Поток сбора трафика запущен")
        else:
            print("Директория для сбора трафика не существует, создайте её и перезапустите процесс!")

    def stop(self):
        print("Поток сбора трафика завершён")
        self.run_sniff = False


class Analyzer:
    def __init__(self, window_size, charact_file_length, charact_file_mask, ip_client, path_name, trffic_name):
        self.window_size            = window_size
        self.charact_file_length    = charact_file_length
        self.charact_file_mask      = charact_file_mask
        self.ip_client              = ip_client
        self.path_name              = path_name
        self.trffic_name            = trffic_name

        self.files_traffic_arr  = []
        self.array_paket_global = []

        self.run_analyz         = True
        self.th_main_analyz     = None
        self.index_charact_file = 0
        self.GetLastFileId()

        self.GetFilesTraffic()

    def GetFilesTraffic(self):
        path_sniffer_home = Path(self.path_name + "\\" + self.trffic_name)
        files_local      = {}
        files_timecreate = []

        for file in path_sniffer_home.iterdir():
            file_split = str(file).split(".")

            if file.is_file():
                if file_split[1] == "pcap" or file_split[1] == "pcapng":
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

        self.files_traffic_arr = []
        if len(files_timecreate) > 0:
            files_timecreate.sort()

            for tc in files_timecreate:
                self.files_traffic_arr.append(files_local[tc])

        return len(self.files_traffic_arr)

    def GetLastFileId(self):
        path_home = Path(self.path_name + "\\" + self.trffic_name)
        file_arr = []

        for file in path_home.iterdir():
            file = str(file)
            if not (".csv" in file):
                continue
            else:
                file_arr.append(file)

        if len(file_arr) > 0:
            indexs_files_pcapng = []

            # Получение индексов файлов с трафиком
            for file_name in file_arr:
                index_file = int(file_name.split(".")[0].split("\\")[-1].split("_")[-1])
                indexs_files_pcapng.append(index_file)

            indexs_files_pcapng.sort()

            self.index_charact_file = indexs_files_pcapng[-1]
        else:
            self.index_charact_file = 0

    def ParseTraffic(self, file_name):
        print(f"Парсинг файла с трафиком: {file_name}")
        pakets_characts = []

        if not Path(file_name).exists():
            logging.exception(f"Ошибка: файла траффика с именем {file_name} не существует!")
            return

        pcap_file = open(file_name, "rb")
        # Определяем какой тим файла будем считывать и считываем его с помощью dpkt
        if file_name[-1] == "g":
            pcap_file_read = dpkt.pcapng.Reader(pcap_file)
        elif file_name[-1] == "p":
            pcap_file_read = dpkt.pcap.Reader(pcap_file)
        else:
            pcap_file_read = []

        # Считывание каждого пакета для дальнейшей распаковки в набор характеристик
        for timestamp, raw in pcap_file_read:
            characts = {"timestamp": timestamp}

            eth = dpkt.ethernet.Ethernet(raw)
            ip = eth.data
            if not isinstance(ip, dpkt.ip.IP):
                try:
                    ip = dpkt.ip.IP(raw)
                except dpkt.dpkt.UnpackError:
                    continue

            seg = ip.data
            try:
                if type(ip_address(ip.src)) is IPv6Address:
                    print("IPv6")
            except ValueError:
                continue

            if isinstance(seg, dpkt.tcp.TCP):
                characts["transp_protocol"] = 1  # Значение 1 это TCP

                if seg.flags & dpkt.tcp.TH_SYN:
                    characts["syn_flag"] = 1
                else:
                    characts["syn_flag"] = 0

                if seg.flags & dpkt.tcp.TH_ACK:
                    characts["ask_flag"] = 1
                else:
                    characts["ask_flag"] = 0

            elif isinstance(seg, dpkt.udp.UDP):
                characts["transp_protocol"] = 0  # Значение 0 это TCP
                characts["syn_flag"] = 0
                characts["ask_flag"] = 0
            else:
                continue

            characts["ip_src"] = ip.src
            characts["ip_dst"] = ip.dst
            characts["port_src"] = seg.sport
            characts["port_dst"] = seg.dport
            characts["size_paket"] = len(seg)

            pakets_characts.append(characts)

        pcap_file.close()
        print("Обработано пакетов: %d" % len(pakets_characts))

        for paket in pakets_characts:
            self.array_paket_global.append(paket)

        path_new = self.path_name + "\\" + self.trffic_name + "\\" + "Обработанные файлы"
        if not Path(path_new).exists():
            Path(path_new).mkdir()
        file_only_name = file_name.split("\\")[-1]
        Path(file_name).rename(path_new + "\\" + file_only_name)

    def ProcessingTraffic(self, pcap_file_name=None):
        if not (pcap_file_name is None):
            print(f"Запускаем обработку файла: {pcap_file_name}")
            self.ParseTraffic(pcap_file_name)

        array_characts = []
        try:
            while len(self.array_paket_global) >= self.window_size:
                array_pkt = self.array_paket_global[:self.window_size]
                ch = CulcCharactsOnWindow(array_pkt, self.window_size, self.ip_client)
                if ch is not None:
                    array_characts.append(ch)
                else:
                    continue
                self.array_paket_global.pop(0)

        except Exception as err:
            logging.exception(f"Ошибка!\n{err}")
            return False

        print("Выявлено характеристик: %d" % len(array_characts))

        if len(array_characts) == 0:
            print("Не выделено ни одного набора характеристик!")
            return False
        else:
            try:
                characts_file_name = self.path_name + "\\" + self.trffic_name + "\\" + \
                                     self.charact_file_mask + str(self.index_charact_file) + ".csv"

                pd_characts_old = pd.read_csv(characts_file_name)
                pd_characts = pd.DataFrame(array_characts)

                pd_characts_new = pd.concat([pd_characts_old, pd_characts], ignore_index=False)
                pd_characts_arr = []
                num_chunks = math.ceil(len(pd_characts_new) / self.charact_file_length)
                for i in range(num_chunks):
                    pd_characts_arr.append(pd_characts_new[i * self.charact_file_length:(i + 1) * self.charact_file_length])

                pd_characts_arr[0].to_csv(characts_file_name, index=False)
                if len(pd_characts_arr) == 2:
                    self.index_charact_file += 1
                    characts_file_name = self.path_name + "\\" + self.trffic_name + "\\" + \
                                         self.charact_file_mask + str(self.index_charact_file) + ".csv"
                    pd_characts_arr[1].to_csv(characts_file_name, index=False)

                print("Парсинг завершился!")
                return True
            except Exception as err:
                logging.exception(f"Ошибка!\n{err}")
                return False

    def PaketsAnalyz(self, pakets):
        for pkt in pakets:
            self.array_paket_global.append(pkt)

        if Path(self.path_name + "\\" + self.trffic_name).exists():
            characts_file_name = self.path_name + "\\" + self.trffic_name + "\\" + \
                                 self.charact_file_mask + str(self.index_charact_file) + ".csv"
            if not Path(characts_file_name).exists():
                pd_ch_name = pd.DataFrame()
                for ch in CHARACTERISTIC:
                    pd_ch_name[ch] = []
                pd_ch_name.to_csv(str(characts_file_name), index=False)

            print("Запущен анализ заданных пакетов")
            self.ProcessingTraffic()
        else:
            print("Директория с трафиком для анализа не существует, видимо процесс сбора трафика не был запущен ранее!")

    def AnalyzLoop(self):

        while self.run_analyz:
            count_file_traffic = self.GetFilesTraffic()
            if count_file_traffic == 0:
                time.sleep(30)
                continue
            else:
                try:
                    self.ProcessingTraffic(self.files_traffic_arr[0])
                    self.files_traffic_arr.pop(0)
                except IndexError:
                    continue

    def run(self):
        if Path(self.path_name + "\\" + self.trffic_name).exists():
            characts_file_name = self.path_name + "\\" + self.trffic_name + "\\" + \
                                 self.charact_file_mask + str(self.index_charact_file) + ".csv"
            if not Path(characts_file_name).exists():
                pd_ch_name = pd.DataFrame()
                for ch in CHARACTERISTIC:
                    pd_ch_name[ch] = []
                pd_ch_name.to_csv(str(characts_file_name), index=False)

            self.run_analyz = True
            self.th_main_analyz = Thread(target=self.AnalyzLoop, args=())
            self.th_main_analyz.start()
            print("Поток анализа трафика запущен")

        else:
            print("Директория с трафиком для анализа не существует, видимо процесс сбора трафика не был запущен ранее!")


if __name__ == '__main__':
    # Параметры сборщика трафика
    size_pcap_length            = 10000
    iface_name                  = "VMware_Network_Adapter_VMnet3"
    trffic_file_mask            = "traffic_"
    trffic_name                 = "test_dataset_anomaly"
    path_name                   = "F:\\VNAT\\Mytraffic\\youtube_me\\"

    # Дополнительные параметры анализатора трафика
    window_size = 1000
    charact_file_length = 1000000
    charact_file_name = "dataset_"
    ip_client = [IPv4Address("192.168.10.128")]

    sniffer = Sniffer(size_pcap_length, iface_name, trffic_name, trffic_file_mask, path_name)
    sniffer.run()
    # time.sleep(200)
    # sniffer.stop()

    # analizator = Analyzer(window_size, charact_file_length, charact_file_name, ip_client, path_name, trffic_name)
    # analizator.run()



