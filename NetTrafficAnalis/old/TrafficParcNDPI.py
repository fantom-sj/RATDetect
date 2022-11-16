"""
    Модуль, в котором реализованы функции позволяющие разделить захваченный трафик
    на потоки данных, а также выделить основные статистические метрики и рассчитать
    дополнительные метрики, характеризующие каждый из выделенных потоков данных.
"""
from SnifferAndParcerTraffic.TrafficSniffer import StartSniff, PcapConvertToJSON
from flow_characts import CHARACTERISTIC
from netifaces import interfaces, ifaddresses, AF_INET

import dpkt
import pandas as ps
import numpy as np
import json
import pathlib


number = 264

def parse_flows(file_name):
    """
        Прочитать дамп сетевого трафика, разделить его на потоки
        транспортного уровня
        Аргументы:
            pcapfile - путь к файлу PCAP (строка)
        Возвращает (генерирует):
            Список кортежей вида:
                список Ethernet-фреймов для очередного потока
    """
    my_ip_addr = []
    for ifaceName in interfaces():
        addresses = [i['addr'] for i in ifaddresses(ifaceName).setdefault(AF_INET, [{'addr': 'NULL'}])]
        my_ip_addr.append(' '.join(addresses))

    try:
        with open(file_name + ".json", "r", encoding="utf-8") as file:
            pakets = json.load(file)
    except:
        print("Ошибка чтения JSON файла.\nПопытка его пересоздания...")
        file = pathlib.Path(file_name + ".json")
        file.unlink()
        if PcapConvertToJSON(file_name):
            print("Новый JSON успешно создан. Попытка его чтения...")
            try:
                with open(file_name + ".json", "r", encoding="utf-8") as file:
                    pakets = json.load(file)
            except:
                print("Чтение повторно созданного JSON не удалось." +
                      "\nУдаление файла и перезапуск сбора пакетов...")
                file = pathlib.Path(file_name + ".json")
                file.unlink()
                main()


    flows = {}
    for paket in pakets:
        global transp_proto
        try:
            if paket["_source"]["layers"]["ip"]["ip.proto"] == "6":
                transp_proto = "tcp"
            elif paket["_source"]["layers"]["ip"]["ip.proto"] == "17":
                transp_proto = "udp"
            else:
                continue
        except:
            continue

        try:
            ip_src = paket["_source"]["layers"]["ip"]["ip.src"]
            ip_dst = paket["_source"]["layers"]["ip"]["ip.dst"]

            # if (ip_src not in my_ip_addr) or (ip_dst not in my_ip_addr):
            #     continue
        except:
            continue

        global port_src, port_dst
        try:
            if transp_proto == "tcp":
                port_src = int(paket["_source"]["layers"]["tcp"]["tcp.srcport"])
                port_dst = int(paket["_source"]["layers"]["tcp"]["tcp.dstport"])
            elif transp_proto == "udp":
                port_src = int(paket["_source"]["layers"]["udp"]["udp.srcport"])
                port_dst = int(paket["_source"]["layers"]["udp"]["udp.dstport"])
            else:
                continue
        except:
            print("Ошибка в пакете: " + paket["_source"]["layers"]["frame"]["frame.time_epoch"])

        ip_src = bytes(map(int, ip_src.split('.')))
        ip_dst = bytes(map(int, ip_dst.split('.')))
        key = (transp_proto, frozenset(((ip_src, port_src), (ip_dst, port_dst))))
        flows[key] = []


    for ts, raw in dpkt.pcapng.Reader(open(file_name + ".pcapng", "rb")):
        eth = dpkt.ethernet.Ethernet(raw)
        ip = eth.data
        if not isinstance(ip, dpkt.ip.IP):
            continue
        seg = ip.data
        if isinstance(seg, dpkt.tcp.TCP):
            transp_proto = "tcp"
        elif isinstance(seg, dpkt.udp.UDP):
            transp_proto = "udp"
        else:
            continue
        key = (transp_proto, frozenset(((ip.src, seg.sport),
                                        (ip.dst, seg.dport))))

        if key not in flows:
            # print(repr(ip.src))
            continue

        flows[key].append(eth)

    file = pathlib.Path(file_name + ".json")
    file.unlink()

    for key, flow in flows.items():
        yield flow


def forge_flow_stats(flow, strip=0):
    """
        Рассчитать статистические метрики потока.
        Аргументы:
            flow - список Ethernet-фреймов
            strip - количество первых фреймов, по которым
                строить таблицу признаков (если меньше 1,
                то фреймы не отбрасываются)
        Возвращает:
            Словарь, в котором ключи - названия метрик,
            значения - значения этих метрик.
            Если в потоке нет хотя бы двух порций данных,
            возвращает None.
    """

    ip = flow[0].data
    seg = ip.data
    if isinstance(seg, dpkt.tcp.TCP):
        # Смотрим, чтобы в первых двух пакетах был флаг SYN:
        try:
            seg2 = flow[1].data.data
        except IndexError:
            return None
        if not (seg.flags & dpkt.tcp.TH_SYN and seg2.flags & dpkt.tcp.TH_SYN):
            return None
        proto = "tcp"
        flow = flow[3:]  # срезаем tcp handshake
    elif isinstance(seg, dpkt.udp.UDP):
        proto = "udp"
    else:
        raise ValueError("Неизвестный транспортный протокол: `{}`".format(
            seg.__class__.__name__))

    if strip > 0:
        flow = flow[:strip]

    client = (ip.src, seg.sport)
    server = (ip.dst, seg.dport)

    client_portion = []
    server_portion = []
    client_packets = []
    server_packets = []

    cur_portion_size = 0
    cur_portion_owner = "client"
    client_fin = False
    server_fin = False
    for eth in flow:
        ip = eth.data
        seg = ip.data
        if (ip.src, seg.sport) == client:
            if client_fin: continue
            if proto == "tcp":
                client_fin = bool(seg.flags & dpkt.tcp.TH_FIN)
            client_packets.append(len(seg))
            if cur_portion_owner == "client":
                cur_portion_size += len(seg.data)
            elif len(seg.data) > 0:
                server_portion.append(cur_portion_size)
                cur_portion_owner = "client"
                cur_portion_size = len(seg.data)
        elif (ip.src, seg.sport) == server:
            if server_fin: continue
            if proto == "tcp":
                server_fin = bool(seg.flags & dpkt.tcp.TH_FIN)
            server_packets.append(len(seg))
            if cur_portion_owner == "server":
                cur_portion_size += len(seg.data)
            elif len(seg.data) > 0:
                client_portion.append(cur_portion_size)
                cur_portion_owner = "server"
                cur_portion_size = len(seg.data)
        else:
            raise ValueError("There is more than one flow here!")

    if cur_portion_owner == "client":
        client_portion.append(cur_portion_size)
    else:
        server_portion.append(cur_portion_size)

    stats = {
        "client_portion_1": client_portion[0] if len(client_portion) > 0 else 0,
        "client_portion_2": server_portion[0] if len(server_portion) > 0 else 0,
        "server_portion_1": client_portion[1] if len(client_portion) > 1 else 0,
        "server_portion_2": server_portion[1] if len(server_portion) > 1 else 0,

        "client_packet_1": client_packets[0] if len(client_packets) > 0 else 0,
        "client_packet_2": client_packets[1] if len(client_packets) > 1 else 0,
        "server_packet_0": server_packets[0] if len(server_packets) > 0 else 0,
        "server_packet_1": server_packets[1] if len(server_packets) > 1 else 0,
    }

    if client_portion and client_portion[0] == 0:
        client_portion = client_portion[1:]

    if not client_portion or not server_portion:
        return None

    stats.update({
        "transp_proto": int(proto == "tcp"),

        "client_portion_size_avg": np.mean(client_portion),
        "client_portion_size_dev": np.std(client_portion),
        "server_portion_size_avg": np.mean(server_portion),
        "server_portion_size_dev": np.std(server_portion),

        "client_packet_size_avg": np.mean(client_packets),
        "client_packet_size_dev": np.std(client_packets),
        "server_packet_size_avg": np.mean(server_packets),
        "server_packet_size_dev": np.std(server_packets),

        "client_packets_per_portion": len(client_packets) / len(client_portion),
        "server_packets_per_portion": len(server_packets) / len(server_portion),

        "client_effeciency": sum(client_portion) / sum(client_packets),
        "server_efficiency": sum(server_portion) / sum(server_packets),

        "byte_ratio": sum(client_packets) / sum(server_packets),
        "payload_ratio": sum(client_portion) / sum(server_portion),
        "packet_ratio": len(client_packets) / len(server_packets),

        "client_bytes": sum(client_packets),
        "client_payload": sum(client_portion),
        "client_packets": len(client_packets),
        "client_portion": len(client_portion),

        "server_bytes": sum(server_packets),
        "server_payload": sum(server_portion),
        "server_packets": len(server_packets),
        "server_portion": len(server_portion),
    })
    return stats


def main_sniff_new():
    global number
    file_name_origin = "traffic"
    nachalo = number
    for i in range(nachalo, 10000, 1):
        number = i
        print("Пошёл " + str(i) + " цикл сбора трафика!")
        file_name = file_name_origin + "_" + str(i)
        if StartSniff(5, 10000, file_name) == 1:
            flows = {feature: [] for feature in CHARACTERISTIC}
            for flow in parse_flows(file_name):
                stats = forge_flow_stats(flow, 0)
                if stats:
                    for feature in CHARACTERISTIC:
                        flows[feature].append(stats[feature])
            data = ps.DataFrame(flows)
            print(data)
            data.to_csv(file_name + ".csv", index=False)



def main_analiz_old():
    path = "../data/pcap/"

    traffc_RAT_file = [
        path + "traffic_RAT_NingaliNET/traffic_RAT_NingaliNET_1",
        path + "traffic_RAT_NingaliNET/traffic_RAT_NingaliNET_2",
        path + "traffic_RAT_NingaliNET/traffic_RAT_NingaliNET_3",
        path + "traffic_RAT_NingaliNET/traffic_RAT_NingaliNET_4",
        path + "traffic_RAT_NingaliNET/traffic_RAT_NingaliNET_5",

        path + "traffic_RAT_rabbit_hole/RAT_rabbit_hole_1",
        path + "traffic_RAT_rabbit_hole/RAT_rabbit_hole_2",
        path + "traffic_RAT_rabbit_hole/RAT_rabbit_hole_3",
        path + "traffic_RAT_rabbit_hole/RAT_rabbit_hole_4",

        path + "traffic_RAT_revenge/RAT_revenge_1",
        path + "traffic_RAT_revenge/RAT_revenge_2",
        path + "traffic_RAT_revenge/RAT_revenge_3",
        path + "traffic_RAT_revenge/RAT_revenge_4",
        path + "traffic_RAT_revenge/RAT_revenge_5",
        path + "traffic_RAT_revenge/RAT_revenge_6"
    ]

    for file in [path + "traffic_razreshon_1"]:
        PcapConvertToJSON(file)

    for file in [path + "traffic_razreshon_1"]:
        flows = {feature: [] for feature in CHARACTERISTIC}
        for flow in parse_flows(file):
            stats = forge_flow_stats(flow, 0)
            if stats:
                for feature in CHARACTERISTIC:
                    flows[feature].append(stats[feature])
        data = ps.DataFrame(flows)
        print(data)
        data.to_csv(file + ".csv", index=False)


def main():
    try:
        main_sniff_new()
    except Exception as err:
        print("Произошла ошибка при сборе трафика!\n" +
              str(err) + "\n" +
              "Сбор перезапущен.")
        main()


if __name__ == "__main__":
    main()