from pathlib import Path
from ipaddress import IPv4Address, AddressValueError

import pandas as pd
import dpkt


def ExtractionIPKey():
    """
        Метод выявляющий полный список IP адресов, которые фигурируют в
        наборе данных VNAT.
        Возвращает собственно список IP адресов в виде словаря ip_keys:
        {IP_источника - IP_назначения : []}
    """
    # ip_src_arr = []
    # ip_dst_arr = []
    # ip_arr = []
    ip_keys = {}
    VNAT = pd.read_hdf("F:\\VNAT_Dataframe_release_1.h5")
    VNAT = VNAT.to_numpy()
    for flow in VNAT:
        try:
            ip_src = IPv4Address(flow[0][0])
            ip_dst = IPv4Address(flow[0][2])
        except AddressValueError:
            continue
        key = str(ip_src) + " - " + str(ip_dst)
        ip_keys[key] = 0

    #     if ip_src not in ip_src_arr:
    #         ip_src_arr.append(ip_src)
    #
    #
    #     if ip_dst not in ip_dst_arr:
    #         ip_dst_arr.append(ip_dst)
    #
    #     if ip_src not in ip_arr:
    #         ip_arr.append(ip_src)
    #
    #     if ip_dst not in ip_arr:
    #         ip_arr.append(ip_dst)
    #
    # print(ip_src_arr)
    # print(ip_dst_arr)
    # print(ip_arr)

    return ip_keys


def main():
    """
        Модуль для подготовки набора VNAT к обучению, а именно позволяет
        считать все необходимые данные для дальнейшего ручного анализа
        на предмет какие IP адреса имеют клиента в данном наборе.
        Также идёт подсчет сколько каждый источник отправил информации на
        приёмник и запись результата в словарь ip_keys, полученный в ExtractionIPKey().
    """
    dir_VNAT = Path("F:\\VNAT")

    file_pcap_arr = []
    for file in dir_VNAT.iterdir():
        file = str(file)
        file_pcap_arr.append(file)
    print(file_pcap_arr)

    ip_keys = ExtractionIPKey()
    print(ip_keys)
    for pcap_file_str in file_pcap_arr:
        print(pcap_file_str)
        pcap_file = open(pcap_file_str, "rb")

        for timestamp, raw in dpkt.pcap.Reader(pcap_file):
            eth = dpkt.ethernet.Ethernet(raw)
            ip = eth.data
            if not isinstance(ip, dpkt.ip.IP):
                try:
                    ip = dpkt.ip.IP(raw)
                except dpkt.dpkt.UnpackError:
                    continue

            seg = ip.data

            len_paket = len(seg)
            ip_src = IPv4Address(ip.src)
            ip_dst = IPv4Address(ip.dst)

            key = str(ip_src) + " - " + str(ip_dst)
            try:
                ip_keys[key] += len_paket
            except KeyError:
                ip_keys[key] = len_paket

    for key in ip_keys:
        print(key, ":", ip_keys[key])


if __name__ == '__main__':
    main()