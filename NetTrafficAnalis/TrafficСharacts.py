"""
    Модуль содержащий полный набор характеристик сетевого потока,
    а также набор характеристик каждого пакета, необходимых для
    расчёта набора CHARACTERISTIC
"""
import array
from ipaddress import IPv4Address, ip_address, IPv6Address
from pathlib import Path
from enum import Enum
from array import array

import pandas as pd
import numpy as np
import logging
import dpkt

EPOCH_AS_FILETIME       = 116444736000000000  # January 1, 1970 as MS file time
HUNDREDS_OF_NANOSECONDS = 10000000            # Количество наносекунд в одной секунде
MIN_SECOND_INACTIVE     = 5                   # Количество секунд прежде чем поток станет считаться неактивным


class Flow_Charact(Enum):
    # Общие служебные характеристики потока
    Time_Stamp_Start        = 0  # Временная метка начала потока +
    Time_Stamp_End          = 1  # Временная метка конца потока +
    Src_IP_Flow             = 2  # IP-адрес источника сетевого потока +
    Dst_IP_Flow             = 3  # IP-адрес назначения сетевого потока +
    Src_Port_Flow           = 4  # Порт источника сетевого потока +
    Dst_Port_Flow           = 5  # Порт назначения сетевого потока +
    Type_Protocol_Flow      = 6  # Тип протокола сетевого потока (0 - TCP, 1 - UDP) +

    # Статистические характеристики на основе размера сетевого пакета
    Count_Packets_Flow      = 7  # Общее количество пакетов в потоке +
    Count_Fwd_Packets       = 8  # Количество пакетов в направлении от клиента к серверу +
    Count_Bwd_Packets       = 9  # Количество пакетов в направлении от сервера к клиенту +

    Len_Headers_Fwd         = 10  # Общий размер заголовков сетевых пакетов в потоке, в направлении от клиента +
    Len_Fwd_Packets         = 11  # Общий размер пакетов в направлении от клиента к серверу +
    Mean_Len_Fwd_Packets    = 12  # Средний размер пакета в направлении от клиента к серверу +
    Min_Len_Fwd_Packets     = 13  # Минимальный размер пакета в направлении от клиента к серверу +
    Max_Len_Fwd_Packets     = 14  # Максимальный размер пакета в направлении от клиента к серверу +
    Std_Len_Fwd_Packets     = 15  # Стандартное отклонение размера пакета в направлении от клиента к серверу +

    Len_Headers_Bwd         = 16  # Общий размер заголовков сетевых пакетов в потоке, в направлении от сервера +
    Len_Bwd_Packets         = 17  # Общий размер пакетов в направлении от сервера к клиенту +
    Mean_Len_Bwd_Packets    = 18  # Средний размер пакета в направлении от сервера к клиенту +
    Min_Len_Bwd_Packets     = 19  # Минимальный размер пакета в направлении от сервера к клиенту +
    Max_Len_Bwd_Packets     = 20  # Максимальный размер пакета в направлении от сервера к клиенту +
    Std_Len_Bwd_Packets     = 21  # Стандартное отклонение размера пакета в направлении от сервера к клиенту +

    Len_Packets             = 22  # Общий размер всех пакетов в потоке +
    Mean_Len_Packets        = 23  # Средний размер пакета в потоке +
    Min_Len_Packets         = 24  # Минимальный размер пакета в потоке +
    Max_Len_Packets         = 25  # Максимальный размер пакета в потоке +
    Std_Len_Packets         = 26  # Стандартное отклонение размера пакета в потоке +

    # Статистические временные характеристики
    Duration_Flow           = 27  # Длительность потока (разница между Time_Stamp_End и Time_Stamp_Start) +

    Mean_Time_Diff_Pkts     = 28  # Среднее время между пакетами в потоке +
    Min_Time_Diff_Pkts      = 29  # Минимальное время между пакетами в потоке +
    Max_Time_Diff_Pkts      = 30  # Максимальное время между пакетами в потоке +
    Std_Time_Diff_Pkts      = 31  # Стандартное отклонение времени между пакетами в потоке +

    Mean_Time_Diff_Fwd_Pkts = 32  # Среднее время между пакетами в направлении от клиента к серверу +
    Min_Time_Diff_Fwd_Pkts  = 33  # Минимальное время между пакетами в направлении от клиента к серверу +
    Max_Time_Diff_Fwd_Pkts  = 34  # Максимальное время между пакетами в направлении от клиента к серверу +
    Std_Time_Diff_Fwd_Pkts  = 35  # Стандартное отклонение времени между пакетами в направлении от клиента к серверу +

    Mean_Time_Diff_Bwd_Pkts = 36  # Среднее время между пакетами в направлении от сервера к клиенту +
    Min_Time_Diff_Bwd_Pkts  = 37  # Минимальное время между пакетами в направлении от сервера к клиенту +
    Max_Time_Diff_Bwd_Pkts  = 38  # Максимальное время между пакетами в направлении от сервера к клиенту +
    Std_Time_Diff_Bwd_Pkts  = 39  # Стандартное отклонение времени между пакетами в направлении от сервера к клиенту +

    # Характеристики флагов в сетевых пакетах
    Count_Flags_PSH_Fwd     = 40  # Количество пакетов с флагом PSH от клиента к серверу +
    Count_Flags_PSH_Bwd     = 41  # Количество пакетов с флагом PSH от сервера к клиенту +
    Count_Flags_URG_Fwd     = 42  # Количество пакетов с флагом URG от клиента к серверу !!!Не используется!!!
    Count_Flags_URG_Bwd     = 43  # Количество пакетов с флагом URG от сервера к клиенту + !!!Не используется!!!
    Count_Flags_PSH         = 44  # Общее количество пакетов с флагом PSH в сетевом потоке +
    Count_Flags_URG         = 45  # Общее количество пакетов с флагом URG в сетевом потоке + !!!Не используется!!!
    Count_Flags_SYN         = 46  # Общее количество пакетов с флагом SYN в сетевом потоке +
    Count_Flags_ASK         = 47  # Общее количество пакетов с флагом ASK в сетевом потоке +
    Count_Flags_RST         = 48  # Общее количество пакетов с флагом RST в сетевом потоке +
    Count_Flags_FIN         = 49  # Общее количество пакетов с флагом FIN в сетевом потоке +

    # Характеристики скорости передачи данных
    Ratio_Size_Down_UP      = 50  # Соотношение размера отправленных и полученных данных клиентом +
    Speed_Bytes_Fwd         = 51  # Скорость передачи данных в байтах от клиента серверу +
    Speed_Bytes_Bwd         = 52  # Скорость передачи данных в байтах от сервера клиенту +
    Speed_Pkts_Fwd          = 53  # Скорость передачи пакетов от клиента серверу +
    Speed_Pkts_Bwd          = 54  # Скорость передачи пакетов от сервера клиенту +
    Speed_Bytes_Flow        = 55  # Скорость передачи байтов в сетевом потоке +
    Speed_Pkts_Flow         = 56  # Скорость передачи пакетов в сетевом потоке +

    # Характеристики активности сетевого трафика
    Mean_Active_Time_Flow   = 57  # Среднее время активности потока до того как стать неактивным +
    Min_Active_Time_Flow    = 58  # Минимальное время активности потока до того как стать неактивным +
    Max_Active_Time_Flow    = 59  # Максимальное время активности потока до того как стать неактивным +
    Std_Active_Time_Flow    = 60  # Стандартное отклонение времени активности потока до того как стать неактивным +

    Mean_InActive_Time_Flow = 61  # Среднее время не активности потока до того как стать активным +
    Min_InActive_Time_Flow  = 62  # Минимальное время не активности потока до того как стать активным +
    Max_InActive_Time_Flow  = 63  # Максимальное время не активности потока до того как стать активным +
    Std_InActive_Time_Flow  = 64  # Стандартное отклонение времени активности потока до того как стать активным +


class Packet_Charact(Enum):
    timestamp           = 0
    ip_src              = 1
    ip_dst              = 2
    port_src            = 3
    port_dst            = 4
    transp_protocol     = 5
    size_packet         = 6
    size_packet_data    = 7
    size_packet_head    = 8
    psh_flag            = 9
    urg_flag            = 10
    syn_flag            = 11
    ask_flag            = 12
    rst_flag            = 13
    fin_flag            = 14


def ParseTraffic(file_name):
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
        characts = {Packet_Charact.timestamp: round(timestamp * HUNDREDS_OF_NANOSECONDS + EPOCH_AS_FILETIME)}

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
            characts[Packet_Charact.transp_protocol] = 1  # Значение 1 это TCP

            if seg.flags & dpkt.tcp.TH_PUSH:
                characts[Packet_Charact.psh_flag] = 1
            else:
                characts[Packet_Charact.psh_flag] = 0

            if seg.flags & dpkt.tcp.TH_URG:
                characts[Packet_Charact.urg_flag] = 1
            else:
                characts[Packet_Charact.urg_flag] = 0

            if seg.flags & dpkt.tcp.TH_SYN:
                characts[Packet_Charact.syn_flag] = 1
            else:
                characts[Packet_Charact.syn_flag] = 0

            if seg.flags & dpkt.tcp.TH_ACK:
                characts[Packet_Charact.ask_flag] = 1
            else:
                characts[Packet_Charact.ask_flag] = 0

            if seg.flags & dpkt.tcp.TH_RST:
                characts[Packet_Charact.rst_flag] = 1
            else:
                characts[Packet_Charact.rst_flag] = 0

            if seg.flags & dpkt.tcp.TH_FIN:
                characts[Packet_Charact.fin_flag] = 1
            else:
                characts[Packet_Charact.fin_flag] = 0

        elif isinstance(seg, dpkt.udp.UDP):
            characts[Packet_Charact.transp_protocol] = 0  # Значение 0 это TCP
            characts[Packet_Charact.psh_flag] = 0
            characts[Packet_Charact.urg_flag] = 0
            characts[Packet_Charact.syn_flag] = 0
            characts[Packet_Charact.ask_flag] = 0
            characts[Packet_Charact.rst_flag] = 0
            characts[Packet_Charact.fin_flag] = 0
        else:
            continue

        characts[Packet_Charact.ip_src] = int.from_bytes(ip.src, byteorder="big")
        characts[Packet_Charact.ip_dst] = int.from_bytes(ip.dst, byteorder="big")
        characts[Packet_Charact.port_src] = seg.sport
        characts[Packet_Charact.port_dst] = seg.dport
        characts[Packet_Charact.size_packet] = len(raw)
        characts[Packet_Charact.size_packet_data] = len(seg.data)
        characts[Packet_Charact.size_packet_head] = characts[Packet_Charact.size_packet] - \
                                                    characts[Packet_Charact.size_packet_data]

        pakets_characts.append(characts)

    pcap_file.close()

    for pkt in pakets_characts:
        yield pkt


def CulcCharactsFlowOnWindow(array_paket_flow: list, ip_client: list):
    """
        Функция предназначена для расчёта метрик при анализе очередного сетевого
        пакета, считанного из pcapng файла функцией-генератором ParsePcapng.
        Расчёт метрик ведётся в рамках заданного окна.

        Принимает:
            array_paket_flow - массив с характеристиками сетевых пакетов в анализируемом потоке;
            ip_client   - массив с ip адресами клиента, по котором будет определиться
                          кто выступает клиентом в каждом сетевом пакете;
        Возвращает:
            characts_on_window - Статистическую метрику типа CHARACTERISTIC,
                                 описывающую состояние заданного окна в рамках которого идут расчёты.
    """

    characts_flow = {
        Flow_Charact.Time_Stamp_Start:   array_paket_flow[0][Packet_Charact.timestamp],
        Flow_Charact.Time_Stamp_End:     array_paket_flow[-1][Packet_Charact.timestamp],
        Flow_Charact.Src_IP_Flow:        array_paket_flow[0][Packet_Charact.ip_src],
        Flow_Charact.Dst_IP_Flow:        array_paket_flow[0][Packet_Charact.ip_dst],
        Flow_Charact.Src_Port_Flow:      array_paket_flow[0][Packet_Charact.port_src],
        Flow_Charact.Dst_Port_Flow:      array_paket_flow[0][Packet_Charact.port_dst],
        Flow_Charact.Type_Protocol_Flow: array_paket_flow[0][Packet_Charact.transp_protocol],

        Flow_Charact.Count_Packets_Flow: len(array_paket_flow),
        Flow_Charact.Count_Fwd_Packets:  0,
        Flow_Charact.Count_Bwd_Packets:  0,
        Flow_Charact.Len_Headers_Fwd:    0,
        Flow_Charact.Len_Fwd_Packets:    0,
        Flow_Charact.Len_Headers_Bwd:    0,
        Flow_Charact.Len_Bwd_Packets:    0,
        Flow_Charact.Len_Packets:        0,

        Flow_Charact.Duration_Flow: array_paket_flow[-1][Packet_Charact.timestamp] -
                                    array_paket_flow[0][Packet_Charact.timestamp],

        Flow_Charact.Count_Flags_PSH_Fwd: 0,
        Flow_Charact.Count_Flags_PSH_Bwd: 0,
        Flow_Charact.Count_Flags_URG_Fwd: 0,
        Flow_Charact.Count_Flags_URG_Bwd: 0,
        Flow_Charact.Count_Flags_PSH:     0,
        Flow_Charact.Count_Flags_URG:     0,
        Flow_Charact.Count_Flags_SYN:     0,
        Flow_Charact.Count_Flags_ASK:     0,
        Flow_Charact.Count_Flags_RST:     0,
        Flow_Charact.Count_Flags_FIN:     0,
    }

    arr_len_fwd_packets = array("I")
    arr_len_bwd_packets = array("I")
    arr_len_packets     = array("I")

    arr_time_diff_pkts     = array("Q")
    arr_time_diff_fwd_pkts = array("Q")
    arr_time_diff_bwd_pkts = array("Q")
    arr_time_active_flow   = array("Q")
    arr_time_inactive_flow = array("Q")

    pkt_old     = None
    pkt_fwd_old = None
    pkt_bwd_old = None

    start_active = array_paket_flow[0][Packet_Charact.timestamp]

    for pkt in array_paket_flow:
        if not pkt_old is None:
            try:
                arr_time_diff_pkts.append(pkt[Packet_Charact.timestamp] - pkt_old[Packet_Charact.timestamp])
            except:
                continue

            timeout_inactive = pkt[Packet_Charact.timestamp] - pkt_old[Packet_Charact.timestamp]
            if timeout_inactive >= (HUNDREDS_OF_NANOSECONDS * MIN_SECOND_INACTIVE):
                arr_time_inactive_flow.append(timeout_inactive)
                arr_time_active_flow.append(pkt_old[Packet_Charact.timestamp] - start_active)
                start_active = pkt[Packet_Charact.timestamp]

        if pkt[Packet_Charact.ip_src] in ip_client:
            characts_flow[Flow_Charact.Count_Fwd_Packets] += 1

            characts_flow[Flow_Charact.Len_Headers_Fwd] += pkt[Packet_Charact.size_packet_head]
            characts_flow[Flow_Charact.Len_Fwd_Packets] += pkt[Packet_Charact.size_packet]

            arr_len_fwd_packets.append(pkt[Packet_Charact.size_packet])

            if not pkt_fwd_old is None:
                arr_time_diff_fwd_pkts.append(pkt[Packet_Charact.timestamp] - pkt_fwd_old[Packet_Charact.timestamp])
            pkt_fwd_old = pkt

            if pkt[Packet_Charact.psh_flag]:
                characts_flow[Flow_Charact.Count_Flags_PSH_Fwd] += 1
            if pkt[Packet_Charact.urg_flag]:
                characts_flow[Flow_Charact.Count_Flags_URG_Fwd] += 1
        else:
            characts_flow[Flow_Charact.Count_Bwd_Packets] += 1

            characts_flow[Flow_Charact.Len_Headers_Bwd] += pkt[Packet_Charact.size_packet_head]
            characts_flow[Flow_Charact.Len_Bwd_Packets] += pkt[Packet_Charact.size_packet]

            arr_len_bwd_packets.append(pkt[Packet_Charact.size_packet])

            if not pkt_bwd_old is None:
                arr_time_diff_bwd_pkts.append(pkt[Packet_Charact.timestamp] - pkt_bwd_old[Packet_Charact.timestamp])
            pkt_bwd_old = pkt

            if pkt[Packet_Charact.psh_flag]:
                characts_flow[Flow_Charact.Count_Flags_PSH_Bwd] += 1
            if pkt[Packet_Charact.urg_flag]:
                characts_flow[Flow_Charact.Count_Flags_URG_Bwd] += 1

        characts_flow[Flow_Charact.Len_Packets] += pkt[Packet_Charact.size_packet]
        arr_len_packets.append(pkt[Packet_Charact.size_packet])

        pkt_old = pkt

        if pkt[Packet_Charact.psh_flag]:
            characts_flow[Flow_Charact.Count_Flags_PSH] += 1
        if pkt[Packet_Charact.urg_flag]:
            characts_flow[Flow_Charact.Count_Flags_URG] += 1
        if pkt[Packet_Charact.syn_flag]:
            characts_flow[Flow_Charact.Count_Flags_SYN] += 1
        if pkt[Packet_Charact.ask_flag]:
            characts_flow[Flow_Charact.Count_Flags_ASK] += 1
        if pkt[Packet_Charact.rst_flag]:
            characts_flow[Flow_Charact.Count_Flags_RST] += 1
        if pkt[Packet_Charact.fin_flag]:
            characts_flow[Flow_Charact.Count_Flags_FIN] += 1

    if len(arr_len_fwd_packets) == 0:
        characts_flow[Flow_Charact.Mean_Len_Fwd_Packets] = 0
        characts_flow[Flow_Charact.Min_Len_Fwd_Packets]  = 0
        characts_flow[Flow_Charact.Max_Len_Fwd_Packets]  = 0
        characts_flow[Flow_Charact.Std_Len_Fwd_Packets]  = 0
    else:
        characts_flow[Flow_Charact.Mean_Len_Fwd_Packets] = np.mean(arr_len_fwd_packets)
        characts_flow[Flow_Charact.Min_Len_Fwd_Packets]  = np.min(arr_len_fwd_packets)
        characts_flow[Flow_Charact.Max_Len_Fwd_Packets]  = np.max(arr_len_fwd_packets)
        characts_flow[Flow_Charact.Std_Len_Fwd_Packets]  = np.std(arr_len_fwd_packets)

    if len(arr_len_bwd_packets) == 0:
        characts_flow[Flow_Charact.Mean_Len_Bwd_Packets] = 0
        characts_flow[Flow_Charact.Min_Len_Bwd_Packets]  = 0
        characts_flow[Flow_Charact.Max_Len_Bwd_Packets]  = 0
        characts_flow[Flow_Charact.Std_Len_Bwd_Packets]  = 0
    else:
        characts_flow[Flow_Charact.Mean_Len_Bwd_Packets] = np.mean(arr_len_bwd_packets)
        characts_flow[Flow_Charact.Min_Len_Bwd_Packets]  = np.min(arr_len_bwd_packets)
        characts_flow[Flow_Charact.Max_Len_Bwd_Packets]  = np.max(arr_len_bwd_packets)
        characts_flow[Flow_Charact.Std_Len_Bwd_Packets]  = np.std(arr_len_bwd_packets)

    if len(arr_len_packets) == 0:
        characts_flow[Flow_Charact.Mean_Len_Packets] = 0
        characts_flow[Flow_Charact.Min_Len_Packets]  = 0
        characts_flow[Flow_Charact.Max_Len_Packets]  = 0
        characts_flow[Flow_Charact.Std_Len_Packets]  = 0
    else:
        characts_flow[Flow_Charact.Mean_Len_Packets] = np.mean(arr_len_packets)
        characts_flow[Flow_Charact.Min_Len_Packets]  = np.min(arr_len_packets)
        characts_flow[Flow_Charact.Max_Len_Packets]  = np.max(arr_len_packets)
        characts_flow[Flow_Charact.Std_Len_Packets]  = np.std(arr_len_packets)

    if len(arr_time_diff_pkts) == 0:
        characts_flow[Flow_Charact.Mean_Time_Diff_Pkts] = 0
        characts_flow[Flow_Charact.Min_Time_Diff_Pkts]  = 0
        characts_flow[Flow_Charact.Max_Time_Diff_Pkts]  = 0
        characts_flow[Flow_Charact.Std_Time_Diff_Pkts]  = 0
    else:
        characts_flow[Flow_Charact.Mean_Time_Diff_Pkts] = np.mean(arr_time_diff_pkts)
        characts_flow[Flow_Charact.Min_Time_Diff_Pkts]  = np.min(arr_time_diff_pkts)
        characts_flow[Flow_Charact.Max_Time_Diff_Pkts]  = np.max(arr_time_diff_pkts)
        characts_flow[Flow_Charact.Std_Time_Diff_Pkts]  = np.std(arr_time_diff_pkts)

    if len(arr_time_diff_fwd_pkts) == 0:
        characts_flow[Flow_Charact.Mean_Time_Diff_Fwd_Pkts] = 0
        characts_flow[Flow_Charact.Min_Time_Diff_Fwd_Pkts]  = 0
        characts_flow[Flow_Charact.Max_Time_Diff_Fwd_Pkts]  = 0
        characts_flow[Flow_Charact.Std_Time_Diff_Fwd_Pkts]  = 0
    else:
        characts_flow[Flow_Charact.Mean_Time_Diff_Fwd_Pkts] = np.mean(arr_time_diff_fwd_pkts)
        characts_flow[Flow_Charact.Min_Time_Diff_Fwd_Pkts]  = np.min(arr_time_diff_fwd_pkts)
        characts_flow[Flow_Charact.Max_Time_Diff_Fwd_Pkts]  = np.max(arr_time_diff_fwd_pkts)
        characts_flow[Flow_Charact.Std_Time_Diff_Fwd_Pkts]  = np.std(arr_time_diff_fwd_pkts)

    if len(arr_time_diff_bwd_pkts) == 0:
        characts_flow[Flow_Charact.Mean_Time_Diff_Bwd_Pkts] = 0
        characts_flow[Flow_Charact.Min_Time_Diff_Bwd_Pkts]  = 0
        characts_flow[Flow_Charact.Max_Time_Diff_Bwd_Pkts]  = 0
        characts_flow[Flow_Charact.Std_Time_Diff_Bwd_Pkts]  = 0
    else:
        characts_flow[Flow_Charact.Mean_Time_Diff_Bwd_Pkts] = np.mean(arr_time_diff_bwd_pkts)
        characts_flow[Flow_Charact.Min_Time_Diff_Bwd_Pkts]  = np.min(arr_time_diff_bwd_pkts)
        characts_flow[Flow_Charact.Max_Time_Diff_Bwd_Pkts]  = np.max(arr_time_diff_bwd_pkts)
        characts_flow[Flow_Charact.Std_Time_Diff_Bwd_Pkts]  = np.std(arr_time_diff_bwd_pkts)

    if characts_flow[Flow_Charact.Len_Bwd_Packets] != 0:
        characts_flow[Flow_Charact.Ratio_Size_Down_UP] = characts_flow[Flow_Charact.Len_Fwd_Packets] / \
                                                         characts_flow[Flow_Charact.Len_Bwd_Packets]
    else:
        characts_flow[Flow_Charact.Ratio_Size_Down_UP] = np.inf

    if characts_flow[Flow_Charact.Duration_Flow] != 0:
        characts_flow[Flow_Charact.Speed_Bytes_Fwd]  = characts_flow[Flow_Charact.Len_Fwd_Packets] / \
                                                       characts_flow[Flow_Charact.Duration_Flow]
        characts_flow[Flow_Charact.Speed_Bytes_Bwd]  = characts_flow[Flow_Charact.Len_Bwd_Packets] / \
                                                       characts_flow[Flow_Charact.Duration_Flow]
        characts_flow[Flow_Charact.Speed_Pkts_Fwd]   = characts_flow[Flow_Charact.Count_Fwd_Packets] / \
                                                       characts_flow[Flow_Charact.Duration_Flow]
        characts_flow[Flow_Charact.Speed_Pkts_Bwd]   = characts_flow[Flow_Charact.Count_Bwd_Packets] / \
                                                       characts_flow[Flow_Charact.Duration_Flow]
        characts_flow[Flow_Charact.Speed_Bytes_Flow] = characts_flow[Flow_Charact.Len_Packets] / \
                                                       characts_flow[Flow_Charact.Duration_Flow]
        characts_flow[Flow_Charact.Speed_Pkts_Flow]  = characts_flow[Flow_Charact.Count_Packets_Flow] / \
                                                       characts_flow[Flow_Charact.Duration_Flow]
    else:
        characts_flow[Flow_Charact.Speed_Bytes_Fwd]  = np.inf
        characts_flow[Flow_Charact.Speed_Bytes_Bwd]  = np.inf
        characts_flow[Flow_Charact.Speed_Pkts_Fwd]   = np.inf
        characts_flow[Flow_Charact.Speed_Pkts_Bwd]   = np.inf
        characts_flow[Flow_Charact.Speed_Bytes_Flow] = np.inf
        characts_flow[Flow_Charact.Speed_Pkts_Flow]  = np.inf

    if len(arr_time_active_flow) == 0:
        characts_flow[Flow_Charact.Mean_Active_Time_Flow] = 0
        characts_flow[Flow_Charact.Min_Active_Time_Flow]  = 0
        characts_flow[Flow_Charact.Max_Active_Time_Flow]  = 0
        characts_flow[Flow_Charact.Std_Active_Time_Flow]  = 0
    else:
        characts_flow[Flow_Charact.Mean_Active_Time_Flow] = np.mean(arr_time_active_flow)
        characts_flow[Flow_Charact.Min_Active_Time_Flow]  = np.min(arr_time_active_flow)
        characts_flow[Flow_Charact.Max_Active_Time_Flow]  = np.max(arr_time_active_flow)
        characts_flow[Flow_Charact.Std_Active_Time_Flow]  = np.std(arr_time_active_flow)

    if len(arr_time_inactive_flow) == 0:
        characts_flow[Flow_Charact.Mean_InActive_Time_Flow] = 0
        characts_flow[Flow_Charact.Min_InActive_Time_Flow]  = 0
        characts_flow[Flow_Charact.Max_InActive_Time_Flow]  = 0
        characts_flow[Flow_Charact.Std_InActive_Time_Flow]  = 0
    else:
        characts_flow[Flow_Charact.Mean_InActive_Time_Flow] = np.mean(arr_time_inactive_flow)
        characts_flow[Flow_Charact.Min_InActive_Time_Flow]  = np.min(arr_time_inactive_flow)
        characts_flow[Flow_Charact.Max_InActive_Time_Flow]  = np.max(arr_time_inactive_flow)
        characts_flow[Flow_Charact.Std_InActive_Time_Flow]  = np.std(arr_time_inactive_flow)

    return characts_flow


if __name__ == '__main__':
    from timeit import timeit

    traffic_file = "D:\\Пользователи\\Admin\\Рабочий стол\\Статья по КБ\\RATDetect\\data\\pcap\\traffic_RAT_NingaliNET\\traffic_RAT_NingaliNET_1_filter.pcapng"
    ip_client = [int.from_bytes(IPv4Address("192.168.10.128").packed, byteorder="big")]
    arr_pkts = []

    for i in range(1):
        for pkt in ParseTraffic(traffic_file):
            arr_pkts.append(pkt)

    arr_pkts = pd.DataFrame(arr_pkts)
    arr_pkts = arr_pkts.sort_values(Packet_Charact.timestamp)
    arr_pkts = arr_pkts.to_dict("records")
    print(len(arr_pkts))

    timeBool = timeit("""
CulcCharactsFlowOnWindow(arr_pkts, ip_client)
    """, globals=locals(), number=1)
    print(timeBool)

    characts_flow = CulcCharactsFlowOnWindow(arr_pkts, ip_client)
    print(characts_flow)
    characts_flow = pd.DataFrame([characts_flow])
    print(characts_flow)


