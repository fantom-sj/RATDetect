import pandas as pd

from ipaddress import IPv4Address
import matplotlib.pyplot as plt


def main():
    test_dataset    = "..\\data\\pcap\\traffic_test_RAT_and_NoRAT.pcapng"
    malefactor_ip   = IPv4Address("192.168.10.129")
    victim_ip       = IPv4Address("192.168.10.128")
    ip_client       = [victim_ip]
    window_size     = 1000

    array_paket     = []
    array_characts  = []

    # print("Начинаем выявление характеристик в тестовом наборе данных!")
    # try:
    #     for pkt in ParsePcapng(test_dataset):
    #         if (IPv4Address(pkt["ip_src"]) == victim_ip and IPv4Address(pkt["ip_dst"]) == malefactor_ip) or \
    #                 (IPv4Address(pkt["ip_src"]) == malefactor_ip and IPv4Address(pkt["ip_dst"]) == victim_ip):
    #             pkt["metka"] = "RAT"
    #         else:
    #             pkt["metka"] = "NoRAT"
    #
    #         array_paket.append(pkt)
    #         if len(array_paket) == window_size + 1:
    #             array_paket.pop(0)
    #             ch = CulcCharactsOnWindow(array_paket, window_size, ip_client)
    #
    #             RAT_count = 0
    #             for paket in array_paket:
    #                 if paket["metka"] == "RAT":
    #                     RAT_count += 1
    #             ch["RAT_count"] = RAT_count
    #
    #             if ch is not None:
    #                 array_characts.append(ch)
    #             else:
    #                 continue
    #
    # except Exception as err:
    #     logging.exception(f"Ошибка!\n{err}")
    #     return False

    array_characts = pd.read_csv("array_characts.csv").to_dict("records")[:1206000]
    RAT_signature = []

    for idx in range(0, 1206000, 1000):
        ch_srez = array_characts[idx:idx+1000]
        RAT_count_sum = 0
        for ch in ch_srez:
            RAT_count_sum += int(ch["RAT_count"])
        RAT_count_mean = RAT_count_sum/1000
        RAT_normalaz = RAT_count_mean/10
        RAT_signature.append(RAT_normalaz)

    plt.plot(RAT_signature)
    plt.show()

    pd.DataFrame(RAT_signature).to_csv("RAT_signature.csv", index=False)


if __name__ == '__main__':
    main()
