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


def __SniffPackets(iface=None, count=1000, file_name="traffic"):
    jsonfile = open(file_name + '.json', 'w')
    sniffer = ["Wireshark\\tshark.exe", "-T", "json", "-P", "-V", "-x", "-c", str(count), "-w", file_name + ".pcapng"]
    if iface:
        sniffer.append("-i " + str(iface))
    prog = sp.Popen(sniffer, stdin=sp.PIPE, stdout=jsonfile)
    prog.communicate()


def PcapConvertToJSON(file_name):
    jsonfile = open(file_name + '.json', 'w')
    converter = ["Wireshark\\tshark.exe", "-T", "json", "-P", "-V", "-x", "-r", file_name + ".pcapng"]
    try:
        prog = sp.Popen(converter, stdin=sp.PIPE, stdout=jsonfile)
        prog.communicate()
        return True
    except:
        return False


def StartSniff(iface=None, count=1000, file_name="traffic"):
    th_sniff = Thread(target=__SniffPackets, args=(iface, count, file_name,))
    th_sniff.start()
    while th_sniff.is_alive():
        continue
    return True


def main():
    print(StartSniff(4, 25, "traffic"))
    # PcapConvertToJSON("traffic")


if __name__ == '__main__':
    main()