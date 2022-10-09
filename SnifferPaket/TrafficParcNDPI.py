"""
    Модуль, в котором реализованы функции позволяющие разделить захваченный трафик
    на потоки данных, а также выделить основные статистические метрики и рассчитать
    дополнительные метрики, характеризующие каждый из выделенных потоков данных.
"""

from SnifferPaket.TrafficSniffer import GetNumberIface, __SniffPackets
from SnifferPaket.characts import CHARACTERISTIC

from collections import Counter
from ipaddress import IPv4Address, ip_address, IPv6Address
from pathlib import Path
from threading import Thread

import pandas as pd
import numpy as np
import dpkt
import json
import re
import argparse
import logging

array_paket_global = []
preprocessing_queue = []
rat_pkts = []
norat_pkts = []
counter = -50000
add_mode = "norat"

def StartSniffTh(sniffer_home, index_last_file, num_iface, count_pkt_one_pcap):
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
    index_last_file += 1
    while True:
        traffic_file = "{0}\\{1}{2}.pcapng".format(sniffer_home, config_parametrs["pcap_file_name"], str(index_last_file))
        try:
            th_sniff = Thread(target=__SniffPackets, args=(num_iface,
                                                           count_pkt_one_pcap,
                                                           traffic_file,))
            th_sniff.start()
            th_sniff.join()
            preprocessing_queue.append(traffic_file)
            index_last_file += 1
        except Exception as err:
            print("Ошибка во время снифинга! %s"%str(err))
            continue


def ParsePcapng(file_name):
    """
        Функция открывает pcap или pcapng файл и считывает из него
        основные характеристики каждого сетевого пакета согласно
        набору BASE_CHARACTERISTIC, для дальнейшей обработки

        Принимает:
            file_name - имя pcapng без его расширения (до точки)
        Генерирует:
            paket - набор характеристик каждого прочитанного пакета
    """

    pakets_characts = []

    pcap_file = open(file_name, "rb")
    # Определяем какой тим файла будем считывать и считываем его с помощью dpkt
    if file_name[-1] == "g":
        pcap_file_read = dpkt.pcapng.Reader(pcap_file)
    elif file_name[-1] == "p":
        pcap_file_read = dpkt.pcap.Reader(pcap_file)
    else:
        pcap_file_read = []
        yield []

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
    print("Обработано пакетов: %d"%len(pakets_characts))

    for paket in pakets_characts:
        yield paket


def CulcCharactsOnWindow(array_paket, window_size, ip_client):
    """
        Функция предназначена для расчёта метрик при анализе очередного сетевого
        пакета, считанного из pcapng файла функцией-генератором ParsePcapng.
        Расчёт метрик ведётся в рамках заданного окна.

        Принимает:
            array_paket - массив с характеристиками сетевых пакетов длиной в одно окно;
            window_size - размер заданного окна;
            ip_client   - массив с ip адресами клиента, по котором будет определиться
                          кто выступает клиентом в каждом сетевом пакете;
        Возвращает:
            characts_on_window - Статистическую метрику типа CHARACTERISTIC,
                                 описывающую состояние заданного окна в рамках которого идут расчёты.
    """

    arr_ip_src = []
    arr_ip_dst = []
    arr_port_src = []
    arr_port_dst = []
    arr_transp_protocol = []
    arr_size_paket = []

    for pkt in array_paket:
        arr_ip_src.append(IPv4Address(pkt["ip_src"]))
        arr_ip_dst.append(IPv4Address(pkt["ip_dst"]))
        arr_port_src.append(pkt["port_src"])
        arr_port_dst.append(pkt["port_dst"])
        arr_transp_protocol.append(pkt["transp_protocol"])
        arr_size_paket.append(pkt["size_paket"])

    # Устанавливаем временную метку пакета, с приходом которого были рассчитаны характеристики
    Time_Stamp = array_paket[-1]["timestamp"]

    # Посчитываем максимальное количество пакетов с одинаковым IP получателя:
    count_ip_dst = Counter(arr_ip_dst)
    Max_IP_dst_count = max(count_ip_dst.items(), key=lambda c_ip: c_ip[1])[1]

    # Посчитываем максимальное количество пакетов с одинаковым портом источника
    count_port_src = Counter(arr_port_src)
    Max_Port_src_count = max(count_port_src.items(), key=lambda c_port: c_port[1])[1]

    # Посчитываем максимальное количество пакетов с одинаковым портом получателя
    try:
        count_port_dst = Counter(arr_port_dst)
        Max_Port_dst_count = max(count_port_dst.items(), key=lambda c_port: c_port[1])[1]
    except Exception as err:
        Max_Port_dst_count = 0
        # logging.exception(f"Не удалось рассчитать количество пакетов с одинаковым портом получателя!\n{err}")
        # return None

    # Посчитываем количество пакетов переданных по TCP и UDP протоколу
    Count_TCP_pakets = sum(arr_transp_protocol)
    Count_UDP_pakets = window_size - Count_TCP_pakets

    # Посчитываем количество пакетов с одинаковыми портами источника и назначения
    try:
        Count_src_is_dst_ports = len([i for i in zip(arr_port_src, arr_port_dst) if i[0] == i[1]])
    except Exception as err:
        # logging.exception(f"Не удалось рассчитать Count_src_is_dst_ports!\n{err}")
        Count_src_is_dst_ports = 0

    # Посчитываем средний размер и стандартное отклонение среднего размера пакета переданного по протоколу TCP и UDP
    arr_size_paket_TCP = []
    arr_size_paket_UDP = []
    for i in range(window_size):
        if arr_transp_protocol[i]:
            arr_size_paket_TCP.append(arr_size_paket[i])
        else:
            arr_size_paket_UDP.append(arr_size_paket[i])

    try:
        if len(arr_size_paket_TCP) == 0:
            raise ValueError("Размер arr_size_paket_TCP = 0")
        Avg_size_TCP_paket = np.mean(arr_size_paket_TCP)
        Dev_size_TCP_paket = np.std(arr_size_paket_TCP)
    except Exception as err:
        Avg_size_TCP_paket = 0
        Dev_size_TCP_paket = 0
        # logging.exception(f"Не удалось рассчитать Avg_size_TCP_paket и Dev_size_TCP_paket!\n{err}")
        # return None

    try:
        if len(arr_size_paket_UDP) == 0:
            raise ValueError("Размер arr_size_paket_UDP = 0")
        Avg_size_UDP_paket = np.mean(arr_size_paket_UDP)
        Dev_size_UDP_paket = np.std(arr_size_paket_UDP)
    except Exception as err:
        Avg_size_UDP_paket = 0
        Dev_size_UDP_paket = 0
        # logging.exception(f"Не удалось рассчитать Avg_size_UDP_paket и Dev_size_UDP_paket!\n{err}")
        # return None

    # Посчитываем средний размер пакета переданного клиентом и сервером, а также стандартное отклонение
    arr_client_paket_size = []
    arr_server_paket_size = []
    for i in range(window_size):
        if arr_ip_src[i] in ip_client:
            arr_client_paket_size.append(arr_size_paket[i])
        if arr_ip_dst[i] in ip_client:
            arr_server_paket_size.append(arr_size_paket[i])

    try:
        if len(arr_client_paket_size) == 0:
            raise ValueError("Размер arr_client_paket_size = 0")
        Avg_client_paket_size = np.mean(arr_client_paket_size)
        Dev_client_paket_size = np.std(arr_client_paket_size)
    except Exception as err:
        Avg_client_paket_size = 0
        Dev_client_paket_size = 0
        # logging.exception(f"Не удалось рассчитать Avg_client_paket_size и Dev_client_paket_size!\n{err}")
        # return None

    try:
        if len(arr_server_paket_size) == 0:
            raise ValueError("Размер arr_server_paket_size = 0")
        Avg_server_paket_size = np.mean(arr_server_paket_size)
        Dev_server_paket_size = np.std(arr_server_paket_size)
    except Exception as err:
        Avg_server_paket_size = 0
        Dev_server_paket_size = 0
        # logging.exception(f"Не удалось рассчитать Avg_server_paket_size и Dev_server_paket_size!\n{err}")
        # return None

    # Посчитываем количество байт переданных клиентом и сервером в заданном окне, а также их разницу
    Size_client_bytes = sum(arr_client_paket_size)
    Size_server_bytes = sum(arr_server_paket_size)
    Size_difference = abs(Size_client_bytes - Size_server_bytes)

    characts_on_window = {
        "Time_Stamp": Time_Stamp,
        "Max_IP_dst_count": Max_IP_dst_count,
        "Max_Port_src_count": Max_Port_src_count,
        "Max_Port_dst_count": Max_Port_dst_count,

        # "Count_3_way_hs": None,
        # "Count_try_3_way_hs": None,

        "Count_TCP_pakets": Count_TCP_pakets,
        "Count_UDP_pakets": Count_UDP_pakets,
        "Count_src_is_dst_ports": Count_src_is_dst_ports,

        "Avg_size_TCP_paket": Avg_size_TCP_paket,
        "Avg_size_UDP_paket": Avg_size_UDP_paket,

        "Dev_size_TCP_paket": Dev_size_TCP_paket,
        "Dev_size_UDP_paket": Dev_size_UDP_paket,

        "Avg_client_paket_size": Avg_client_paket_size,
        "Avg_server_paket_size": Avg_server_paket_size,

        "Dev_client_paket_size": Dev_client_paket_size,
        "Dev_server_paket_size": Dev_server_paket_size,

        "Size_client_bytes": Size_client_bytes,
        "Size_server_bytes": Size_server_bytes,
        "Size_difference": Size_difference
    }

    return characts_on_window


def PreprocessingPcapng(sniffer_home, window_size, ip_client, pcap_file_name, characts_file_name, rejim):
    """
        Функция предварительной обработки захваченных пакетов и считанных в глобальный массив array_paket.
        Запускает функцию расчёта статистических характеристик пакетов внутри заданного окна,
        после чего сохраняет данные характеристики в csv файл для дальнейшего анализа.
        Принимает:
            sniffer_home - путь к директории сборщика;
            window_size - размер окна, в рамках которого будут вычисляться статистические характеристики;
            ip_client - массив с ip адресами клиента, по котором будет определиться
                        кто выступает клиентом в каждом сетевом пакете (передаётся далее в CulcCharactsOnWindow);
            pcap_file_name - адрес pcapng файла из которого будут считаться пакета для анализа;
            characts_file_name - адрес csv файла куда будут записываться посчитанные характеристики.
        Возвращает:
            True - в случае успешного завершения обработки очередного pcapng файла;
            False - в случае неудачи на одном из этапов вычисления.
    """
    global counter, add_mode
    array_characts = []
    if rejim == 4:
        NoRAT_file, RAT_file = pcap_file_name
        print("Слияние трафка в файлах: ", NoRAT_file, RAT_file)
        try:
            if RAT_file is not None:
                for rat_pkt in ParsePcapng(RAT_file):
                    rat_pkts.append(rat_pkt)
            else:
                print("Добавляем NoRAT трафик")

            for norat_pkt in ParsePcapng(NoRAT_file):
                norat_pkts.append(norat_pkt)

            if len(rat_pkts) == 0:
                array_paket_global.append(norat_pkts.pop(0))
                counter += 1

            else:
                while len(rat_pkts) > 0 and len(norat_pkts) > 0:
                    if add_mode == "norat":
                        array_paket_global.append(norat_pkts.pop(0))
                        counter += 1
                        if counter == 10000:
                            counter = 0
                            add_mode = "rat"
                    elif add_mode == "rat":
                        array_paket_global.append(rat_pkts.pop(0))
                        array_paket_global.append(norat_pkts.pop(0))
                        counter += 1
                        if counter == 2000:
                            counter = 0
                            add_mode = "norat"

            while len(array_paket_global) > window_size:
                array_paket = array_paket_global[0:window_size]
                array_paket_global.pop(0)
                ch = CulcCharactsOnWindow(array_paket, window_size, ip_client)
                if ch is not None:
                    array_characts.append(ch)
                else:
                    continue

        except Exception as err:
            logging.exception(f"Ошибка!\n{err}")
            return False


    else:
        print("Парсинг файла: %s"%pcap_file_name)

        try:
            for pkt in ParsePcapng(pcap_file_name):
                array_paket_global.append(pkt)

                if len(array_paket_global) == window_size + 1:
                    array_paket_global.pop(0)
                    ch = CulcCharactsOnWindow(array_paket_global, window_size, ip_client)
                    if ch is not None:
                        array_characts.append(ch)
                    else:
                        continue
        except Exception as err:
            logging.exception(f"Ошибка!\n{err}")
            return False

    print("Выявлено характеристик: %d" % len(array_characts))

    if len(array_characts) == 0:
        print("Не выделено ни одного набора характеристик!")
        return False
    else:
        try:
            pd_characts_old = pd.read_csv(characts_file_name)
            pd_characts = pd.DataFrame(array_characts)
            pd_characts_new = pd.concat([pd_characts_old, pd_characts], ignore_index=False)
            pd_characts_new.to_csv(characts_file_name, index=False)
            # Path(pcap_file_name).unlink()
            if rejim != 4:
                if Path(str(sniffer_home) + "\\temp").is_dir() is False:
                    Path(str(sniffer_home) + "\\temp").mkdir(mode=0o777, parents=False, exist_ok=False)


                Path(pcap_file_name).rename(str(sniffer_home) + "\\temp\\" + pcap_file_name.split("\\")[-1])

                if rejim == 3:
                    Path(characts_file_name).rename(str(sniffer_home) + "\\temp\\characts_" +
                                                    pcap_file_name.split("\\")[-1].split(".")[0] + ".csv")

                    path_characts_file = Path(characts_file_name)
                    if path_characts_file.exists() is False:
                        pd_ch_name = pd.DataFrame()
                        for ch in CHARACTERISTIC:
                            pd_ch_name[ch] = []
                        pd_ch_name.to_csv(str(path_characts_file), index=False)

            # sp.check_call(["attrib", "+H", pcap_file_name])
            print("Парсинг завершился!")
            return True
        except Exception as err:
            logging.exception(f"Ошибка!\n{err}")
            return False


def GetIdFirstAndLastFl(sniffer_home):
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

    path_sniffer_home = Path(sniffer_home)
    file_arr = []

    for file in path_sniffer_home.iterdir():
        file = str(file)
        if not ("traffic_" in file):
            continue
        else:
            file_arr.append(file)

    if len(file_arr) > 0:
        indexs_files_pcapng = []

        # Получение индексов файлов с трафиком
        for file_name in file_arr:
            index_SnHome = file_name.find("SnHome")
            index_file = [int(s) for s in re.split('_|.p', file_name[index_SnHome + 5:]) if s.isdigit()][0]
            indexs_files_pcapng.append(index_file)

        indexs_files_pcapng.sort()

        for index in indexs_files_pcapng:
            traffic_file = "{0}\\{1}{2}.pcapng".format(sniffer_home, config_parametrs["pcap_file_name"], index)
            preprocessing_queue.append(traffic_file)

        print(indexs_files_pcapng[0], indexs_files_pcapng[-1])
        return indexs_files_pcapng[-1]
    else:
        return 0


def main(config_parametrs, sniffer_home, ip_client, characts_file, rejim):
    """
        Главная функция программы, отвечает за выбор соответствующего заданного режима работы.
        Всего режимов 4:
            0 - запуск снипера и парсера захваченных пакетов;
            1 - запуск только парсера;
            2 - запуск только снифера;
            3 - специальный режим для парсинга датасета пакетов VNAT.
            4 - создаёт см ешенный файл характеристик с трафиком RAT и NoRAT
        Принимает на входе:
            config_parametrs - словарь с конфигурациями программы;
            sniffer_home - домашнюю дирректорию для работы в режимах 0, 1, 2;
            ip_client - массив с IP адресами клиентов;
            characts_file - имя файла для записи характеристик;
            rejim - цифровой режим работы: 0, 1, 2 или 3, 4
    """


    num_iface = GetNumberIface(config_parametrs["iface_name"])
    if num_iface is False:
        print("Неверно задано имя интерфейса, пожалуйста переконфигурируйте файл конфигурации!")
        return 0

    if rejim == 0 or rejim == 2:
        index_last_file = GetIdFirstAndLastFl(sniffer_home)
        print("Запускаем сниффинг!")
        th_sniff = Thread(target=StartSniffTh, args=(sniffer_home,
                                                     index_last_file,
                                                     num_iface,
                                                     config_parametrs["count_pkt_one_pcap"],))
        th_sniff.start()
        if rejim == 2:
            th_sniff.join()

    if rejim == 0 or rejim == 1:
        print("Запускаем парсинг!")
        while True:
            if len(preprocessing_queue) > 0:
                res = PreprocessingPcapng(sniffer_home,
                                          config_parametrs["window_size"],
                                          ip_client, preprocessing_queue[0],
                                          characts_file, rejim)
                if res:
                    print("Парсинг выполнен успешно!\n")
                    preprocessing_queue.pop(0)
                else:
                    print("Парсинг не выполнен!\n")

    if rejim == 3:
        print("Запускаем парсинг VNAT!")
        dir_VNAT = Path("F:\\VNAT\\")

        for file in dir_VNAT.iterdir():
            file = str(file)
            if not ("pcap" in file):
                continue
            else:
                preprocessing_queue.append(file)

        while len(preprocessing_queue) > 0:
            res = PreprocessingPcapng(dir_VNAT,
                                      config_parametrs["window_size"],
                                      ip_client, preprocessing_queue[0],
                                      characts_file, rejim)
            if res:
                print("Парсинг выполнен успешно!\n")
                preprocessing_queue.pop(0)
            else:
                print("Парсинг не выполнен!\n")
                exit(-1)

    if rejim == 4:
        print("Запускаем запускаем смешивание трафика RAT и NoRAT!")
        dir = "..\\data\\pcap\\"

        RAT_file = []
        NoRAT_file = []

        for file in Path(dir + "RAT").iterdir():
            if "NingaliNET" in str(file):
                RAT_file.append(str(file))

        c = 0
        for file in Path("C:\\Users\\Admin\\SnHome\\temp\\NoRAT").iterdir():
            if c == 48:
                break
            NoRAT_file.append(str(file))
            c += 1

        while counter < 100000:
            if len(RAT_file) == 0:
                RAT_file.append(None)

            res = PreprocessingPcapng(dir,
                                      config_parametrs["window_size"],
                                      ip_client, (NoRAT_file[0], RAT_file[0]),
                                      characts_file, rejim)
            if res:
                print("Парсинг выполнен успешно!\n")
                NoRAT_file.pop(0)
                RAT_file.pop(0)
            else:
                print("Парсинг не выполнен!\n")
                exit(-1)

    else:
        print("Не верно задан режим работы")

    return 0


if __name__ == "__main__":
    """
        Стартовая точка программы, где происходит её настройка 
        для дальнейшего запуска в функции main.
        Здесь считывается заданный режим работы и создаются необходимые файлы и каталоги для
        последующей работы в заданном режиме.
    """

    parser = argparse.ArgumentParser()
    parser.add_argument("rejim", help="Введити режим работы (0 - парсинг и снифинг, "
                                      "1 - только парсинг, 2 - только снифинг)", type=str)
    args = parser.parse_args()
    rejim = int(args.__dict__["rejim"])

    home = "..\\data\\pcap\\" # str(Path.home())
    path_sniffer_home = Path(home + "\\temp") # Path(home + "\\SnHome")
    ip_client = [IPv4Address("192.168.0.144"), IPv4Address("192.168.100.75"), IPv4Address("192.168.10.128")]

    if path_sniffer_home.is_dir() is False:
        path_sniffer_home.mkdir(mode=0o777, parents=False, exist_ok=False)
    else:
        print("Папка %s уже существует!" % str(path_sniffer_home))

    path_config_file = Path(str(path_sniffer_home) + "\\config.json")
    if path_config_file.exists() is False:
        path_config_file.touch()
        config_parametrs = {"count_pkt_one_pcap": 10000, "window_size": 1000,
                            "pcap_file_name": "traffic_", "characts_file_name": "characts.csv",
                            "iface_name": "VMware Network Adapter VMnet3"}
        with open(str(path_config_file), 'w') as file_conf:
            json.dump(config_parametrs, file_conf)
            print("настройте программу с помощью файла конфигурации:")
            print(str(path_config_file))
            print("И перезапустите её!")
        file_conf.close()

    else:
        print("Конфиг %s уже существует!" % str(path_config_file))
        with open(str(path_config_file)) as json_file:
            config_parametrs = json.load(json_file)
        json_file.close()

        characts_file = str(path_sniffer_home) + "\\" + config_parametrs["characts_file_name"]
        path_characts_file = Path(characts_file)
        if path_characts_file.exists() is False:
            pd_ch_name = pd.DataFrame()
            for ch in CHARACTERISTIC:
                pd_ch_name[ch] = []
            pd_ch_name.to_csv(str(path_characts_file), index=False)

        # ifaces = get_windows_if_list()
        # for iface in ifaces:
        #     if iface["name"] == config_parametrs["iface_name"]:
        #         print(iface["ips"][-1])
        #         ip_client.append(IPv4Address(iface["ips"][-1]))

        main(config_parametrs, str(path_sniffer_home), ip_client, str(path_characts_file), rejim)



"""
                 IPv4Address("10.101.1.100"), IPv4Address("10.103.1.2"), IPv4Address("10.102.1.2"),
                 IPv4Address("10.104.1.158"), IPv4Address("10.104.1.2"), IPv4Address("10.105.1.2"),
                 IPv4Address("10.113.1.150"), IPv4Address("10.115.1.2"), IPv4Address("10.114.1.1"),
                 IPv4Address("10.115.1.123"), IPv4Address("10.118.1.100"), IPv4Address("10.118.1.2"),
                 IPv4Address("10.116.1.2"), IPv4Address("10.117.1.1"), IPv4Address("10.121.1.130"),
                 IPv4Address("10.121.1.145"), IPv4Address("10.119.1.197"), IPv4Address("10.119.1.197"),
                 IPv4Address("10.119.1.197"), IPv4Address("10.119.1.197"), IPv4Address("10.119.1.197"),
                 IPv4Address("10.119.1.197"), IPv4Address("10.119.1.197"), IPv4Address("10.119.1.197"),
                 IPv4Address("10.119.1.197"), IPv4Address("10.119.1.197"), IPv4Address("10.119.1.197"),
                 IPv4Address("10.119.1.197"), IPv4Address("10.119.1.197"), IPv4Address("10.119.1.197"),
                 IPv4Address("10.119.1.197"), IPv4Address("10.119.1.197"), IPv4Address("10.119.1.197"),
                 IPv4Address("10.119.1.197"), IPv4Address("10.119.1.197"), IPv4Address("10.119.1.197"),
                 IPv4Address("10.119.1.197"), IPv4Address("10.119.1.2"), IPv4Address("10.120.1.1"),
                 IPv4Address("10.120.1.1"), IPv4Address("10.122.1.103"), IPv4Address("10.122.1.2"),
                 IPv4Address("10.123.1.2"), IPv4Address("192.168.0.180"),
"""