"""
    Модуль содержащий полный набор характеристик сетевого потока
"""

CHARACTERISTIC = [
    "transp_proto",  # Транспортный протокол (0 - UDP; 1 - TCP)

    "client_portion_1",  # размер первой порции данных со стороны клиента
    "client_portion_2",  # размер второй порции данных со стороны клиента
    "server_portion_1",  # размер первой порции данных со стороны сервера
    "server_portion_2",  # размер второй порции данных со стороны сервера

    "client_packet_1",  # размер первого сегмента со стороны клиента
    "client_packet_2",  # размер второго сегмента со стороны клиента
    "server_packet_0",  # размер первого сегмента со стороны сервера
    "server_packet_1",  # размер второго сегмента со стороны сервера

    "client_portion_size_avg",  # средный размер порции данных со стороны клиента
    "client_portion_size_dev",  # стандартное отклонение размера порции данных со стороны клиента
    "server_portion_size_avg",  # средный размер порции данных со стороны сервера
    "server_portion_size_dev",  # стандартное отклонение размера порции данных со стороны сервера

    "client_packet_size_avg",  # средный размер сегмента со стороны клиента
    "client_packet_size_dev",  # стандартное отклонение размера сегмента со стороны клиента
    "server_packet_size_avg",  # средний размер сегмента со стороны сервера
    "server_packet_size_dev",  # стандартное отклонение размера сегмента со стороны сервера

    "client_packets_per_portion",  # среднее количество сегментов на порцию данных со стороны клиента
    "server_packets_per_portion",  # среднее количество сегментов на порцию данных со стороны сервера

    "client_effeciency",  # КПД клиента
    "server_efficiency",  # КПД сервера

    "byte_ratio",  # во сколько раз клиент передал больше байт, чем сервер
    "payload_ratio",  # во сколько раз клиент передал больше полезной нагрузки, чем сервер
    "packet_ratio",  # во сколько раз клиент передал больше сегментов, чем сервер

    "client_bytes",    # сколько байт суммарно передано клиентом
    "client_payload",  # сколько полезной нагрузки суммарно передано клиентом
    "client_packets",  # сколько сегментов суммарно передано клиентом
    "client_portion",  # сколько порций данных суммарно передано клиентом

    "server_bytes",  # сколько байт суммарно передано сервером
    "server_payload",  # сколько полезной нагрузки суммарно передано сервером
    "server_packets",  # сколько сегментов суммарно передано сервером
    "server_portion",  # сколько порций данных суммарно передано сервером
]

def Count3WayHsInFlows(array_paket, ip_client):
    flows = {}
    for paket in array_paket:
        key = (paket["transp_protocol"],
               frozenset(((paket["ip_src"], paket["port_src"]),
                          (paket["ip_dst"], paket["port_dst"]))))
        flows[key] = []

    for paket in array_paket:
        key = (paket["transp_protocol"],
               frozenset(((paket["ip_src"], paket["port_src"]),
                          (paket["ip_dst"], paket["port_dst"]))))
        if key not in flows:
            continue
        else:
            flows[key].append(paket)

    Count_3_way_hs = 0
    for key, flow in flows.items():
        try:
            len_flow = len(flow)
            for i in range(len_flow):
                if flow[i]["syn_flag"] and flow[i + 1]["syn_flag"] \
                        and flow[i + 1]["ask_flag"] and flow[i + 2]["ask_flag"]:
                    Count_3_way_hs = + 1
        except:
            continue

    return Count_3_way_hs
