"""
    Данный модель предназначен для запуска процесса снифинга трафика с помощью
    программы tshark, а также последующего перевода захваченного трафика в форму
    json файла с сохранением дампа данных в формате pcapng

    Идеи на будущее:
        1) Заменить снифинг трафика с помощью tshark на снифинг с помощью мадуля scapy
        2) Собирать трафик на прямую в pcap файл и от туда его считывать при анализе и выявлении
           признаков с помощью модуля dpkt
"""

import subprocess as sp
from threading import Thread


def __SniffPackets(iface, count, file_name):
    # jsonfile = open(file_name + '.json', 'w')
    sniffer = ["Wireshark\\tshark.exe", "-c", str(count), "-w", file_name]
    CREATE_NO_WINDOW = 0x08000000
    if iface:
        sniffer.append("-i " + str(iface))
    prog = sp.Popen(sniffer, creationflags=CREATE_NO_WINDOW)
    prog.communicate()


def GetNumberIface(iface_name):
    get_ifaces = ["Wireshark\\tshark.exe", "-D"]
    CREATE_NO_WINDOW = 0x08000000
    prog = sp.Popen(get_ifaces, stdout=sp.PIPE, creationflags=CREATE_NO_WINDOW)
    ifaces, err = prog.communicate()
    for iface in ifaces.decode("utf_8").split("\n"):
        if "(" + iface_name + ")" in iface:
            number_iface = iface[0]
            return number_iface
    return False


def StartSniff(iface, count, file_name):
    try:
        th_sniff = Thread(target=__SniffPackets, args=(iface, count, file_name,))
        th_sniff.start()
        while th_sniff.is_alive():
            continue
        return True
    except:
        return False


def main():
    iface = GetNumberIface("Беспроводная сеть")
    # print(iface)
    # numder = 301
    #
    # while True:
    #     if StartSniff(iface, 10000, "traffic_" + str(numder) + ".pcapng"):
    #         numder += 1
    #     else:
    #         continue


if __name__ == '__main__':
    main()