"""
    Модуль содержащий полный набор характеристик сетевого потока,
    а также набор характеристик каждого пакета, необходимых для
    расчёта набора CHARACTERISTIC
"""
from ipaddress import IPv4Address, ip_address, IPv6Address
from collections import Counter
from pathlib import Path

import numpy as np
import logging
import dpkt

EPOCH_AS_FILETIME = 116444736000000000  # January 1, 1970 as MS file time
HUNDREDS_OF_NANOSECONDS = 10000000

CHARACTERISTIC_FLOW = [
    # Общие характеристики потока
    "Time_Stamp_Start",         # 0. Временная метка начала потока
    "Time_Stamp_End",           # 1. Временная метка конца потока
    "Direction_IP_Port",        # 2. Уникальные направления пакетов в выборке (IP и порт назначения)
    "Max_IP_dst_count",         # 3. Максимальное количество пакетов с одинаковым IP получателя
    "Max_Port_src_count",       # 4. Максимальное количество пакетов с одинаковым портом источника
    "Max_Port_dst_count",       # 5. Максимальное количество пакетов с одинаковым портом получателя
    
    "Count_TCP_pakets",         # 6. Количество пакетов переданных по TCP протоколу
    "Count_UDP_pakets",         # 7. Количество пакетов переданных по UDP протоколу
    "Count_src_is_dst_ports",   # 8. Количество пакетов с одинаковыми портами источника и назначения
    
    "Avg_size_TCP_paket",       # 9. Средний размер пакета переданного по протоколу TCP
    "Avg_size_UDP_paket",       # 10. Средний размер пакета переданного по протоколу UDP

    "Dev_size_TCP_paket",       # 11. Стандартное отклонение размера пакета переданного по протоколу TCP
    "Dev_size_UDP_paket",       # 12. Стандартное отклонение размера пакета переданного по протоколу UDP
    
    "Avg_client_paket_size",    # 13. Средний размер пакета переданного клиентом
    "Avg_server_paket_size",    # 14. Средний размер пакета переданного сервером
    
    "Dev_client_paket_size",    # 15. Стандартное отклонение размера пакета переданного клиентом
    "Dev_server_paket_size",    # 16. Стандартное отклонение размера пакета переданного сервером
    
    "Size_client_bytes",        # 17. Количество байт переданных клиентом в заданном окне
    "Size_server_bytes",        # 18. Количество байт переданных сервером в заданном окне
    "Size_difference",          # 19. Разница размеров переданных данных клиентом и сервером

    "Count_syn_flag",           # 20. Количество пакетов с установленными syn флагами
    "Count_ask_flag",           # 21. Количество пакетов с установленными ask флагами
    "Count_syn_ask_flag"        # 22. Количество пакетов с установленными syn и ask флагами
]

BASE_CHARACTERISTIC = [
    "timestamp",            # Временная метка пакета
    "ip_src",               # IP адрес источника
    "ip_dst",               # IP адрес получателя
    "port_src",             # Порт источника
    "port_dst",             # Порт получателя
    "transp_protocol",      # Транспортный протокол (1 - TCP или 0 - UDP)
    "size_paket",           # Размер пакета
    "syn_flag",             # Наличие флага SYN
    "ask_flag"              # Наличие флага ASK
]


def ParseTraffic(file_name):
    # print(f"Парсинг файла с трафиком: {file_name}")
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
    # print("Обработано пакетов: %d" % len(pakets_characts))

    for pkt in pakets_characts:
        yield pkt


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

    arr_ip_src          = []
    arr_ip_dst          = []
    arr_port_src        = []
    arr_port_dst        = []
    arr_transp_protocol = []
    arr_size_paket      = []
    arr_syn_flag        = []
    arr_ask_flag        = []
    arr_syn_ask_flag    = []

    Direction_IP_Port   = None

    for pkt in array_paket:
        if not IPv4Address(pkt["ip_dst"]) in ip_client:
            direction = str(IPv4Address(pkt["ip_dst"])) + ":" + str(pkt["port_dst"])
            if Direction_IP_Port is None:
                Direction_IP_Port = direction
            elif not direction in Direction_IP_Port:
                Direction_IP_Port += (";" + direction)

        arr_ip_src.append(IPv4Address(pkt["ip_src"]))
        arr_ip_dst.append(IPv4Address(pkt["ip_dst"]))
        arr_port_src.append(pkt["port_src"])
        arr_port_dst.append(pkt["port_dst"])
        arr_transp_protocol.append(pkt["transp_protocol"])
        arr_size_paket.append(pkt["size_paket"])

        if pkt["syn_flag"] == 0 and pkt["ask_flag"] == 1:
            arr_ask_flag.append(1)
        elif pkt["syn_flag"] == 1 and pkt["ask_flag"] == 0:
            arr_syn_flag.append(1)
        elif pkt["syn_flag"] == 1 and pkt["ask_flag"] == 1:
            arr_syn_ask_flag.append(1)

    # Устанавливаем временную метку пакета, с приходом которого были рассчитаны характеристики
    Time_Stamp_Start = round(array_paket[0]["timestamp"] * HUNDREDS_OF_NANOSECONDS + EPOCH_AS_FILETIME)
    Time_Stamp_End   = round(array_paket[-1]["timestamp"] * HUNDREDS_OF_NANOSECONDS + EPOCH_AS_FILETIME)

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

    # Посчитываем количество пакетов с разными комбинациями syn и ask флагов
    Count_syn_flag      = sum(arr_syn_flag)
    Count_ask_flag      = sum(arr_ask_flag)
    Count_syn_ask_flag  = sum(arr_syn_ask_flag)

    characts_on_window = {
        "Time_Stamp_Start": Time_Stamp_Start,
        "Time_Stamp_End": Time_Stamp_End,
        "Direction_IP_Port": Direction_IP_Port,
        "Max_IP_dst_count": Max_IP_dst_count,
        "Max_Port_src_count": Max_Port_src_count,
        "Max_Port_dst_count": Max_Port_dst_count,

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
        "Size_difference": Size_difference,

        "Count_syn_flag": Count_syn_flag,
        "Count_ask_flag": Count_ask_flag,
        "Count_syn_ask_flag": Count_syn_ask_flag,
    }

    return characts_on_window