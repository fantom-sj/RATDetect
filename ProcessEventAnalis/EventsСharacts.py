"""
    Модуль содержащий полный набор характеристик событий процессов в системе,
    а также набор характеристик каждого события, необходимых для
    расчёта набора CHARACTERISTIC_EVENTS
"""
import time

from ipaddress import IPv4Address, AddressValueError
from ProcmonParser.consts import EventClass
from enum import Enum
from socket import gethostbyname, gaierror, getservbyname

import numpy as np
from array import array
import logging
from tqdm import tqdm

HUNDREDS_OF_NANOSECONDS = 10000000            # Количество наносекунд в одной секунде
MIN_SECOND_INACTIVE     = 5                   # Количество секунд прежде чем поток станет считаться неактивным


class Events_Charact(Enum):
    # Общие сведенья о порции событий
    Time_Stamp_Start        = 0  # Временная метка начала порции +
    Time_Stamp_End          = 1  # Временная метка конца порции +
    Process_Name            = 2  # Имя процесса +
    Direction_IP_Port       = 3  # Уникальные направления сетевых инициализаций (IP и порт назначения) +
    Count_Events_Batch      = 4  # Количество событий в порции +

    Count_Success_Events    = 5  # Количество успешных событий +

    # Количество событий разной класса
    Count_Event_Process        = 6  # Количество событий типа Process +
    Count_Event_Registry       = 7  # Количество событий типа Registry +
    Count_Event_File_System    = 8  # Количество событий типа File_System +
    Count_Event_Network        = 9  # Количество событий типа Network +

    # Данные по категории событий: чтение и запись данных процессом
    Count_event_Read           = 10  # Количество событий относящихся к категории чтение +
    Count_event_Read_MetaD     = 11  # Количество событий относящихся к категории чтение метаданных +
    Count_event_Write          = 12  # Количество событий относящихся к категории записи +
    Count_event_Write_MetaD    = 13  # Количество событий относящихся к категории записи метаданных +
    Ratio_Read_on_Write        = 14  # Соотношение количества событий на чтения и записи +

    # Основные события работы с файловой системой
    Count_ReadFile           = 15  # Количество событий на чтение фала +
    Count_CreateFile         = 16  # Количество событий на создание фала +
    Count_WriteFile          = 17  # Количество запросов на запись файла +
    Count_CloseFile          = 18  # Количество запросов на закрытие файла +
    Ratio_Read_on_Write_File = 19  # Соотношение количество событий чтения и записи файлов +

    # Статистические характеристики относящиеся к работе с файловой системой
    Count_Unique_Path  = 45  # Количество уникальных путей обращения к файлам +
    Count_Read_Length  = 46  # Количество считанной с диска информации +
    Count_Write_Length = 47  # Количество записанной на диск информации +

    Mean_Read_Length   = 48  # Средний объем считанной с диска информации в рамках одного события +
    Max_Read_Length    = 49  # Максимальный объем считанной с диска информации в рамках одного события +
    Min_Read_Length    = 50  # Минимальный объем считанной с диска информации в рамках одного события +
    Std_Read_Length    = 51  # Стандартное отклонение объема считанной с диска информации в рамках одного события +

    Mean_Write_Length  = 52  # Средний объем записанной на диск информации в рамках одного события +
    Max_Write_Length   = 53  # Максимальный объем записанной на диск информации в рамках одного события +
    Min_Write_Length   = 54  # Минимальный объем записанной на диск информации в рамках одного события +
    Std_Write_Length   = 55  # Стандартное отклонение объема записанной на диск информации в рамках одного события +

    # Скорость чтения / записи и другие временные характеристики
    Speed_Read_Data          = 56  # Скорость чтения данных +
    Speed_Write_Data         = 57  # Скорость записи данных +

    Mean_Active_Time_Process = 58  # Среднее время активности процесса +
    Max_Active_Time_Process  = 59  # Максимальное время активности процесса +
    Min_Active_Time_Process  = 60  # Минимальное время активности процесса +
    Std_Active_Time_Process  = 61  # Стандартное отклонение времени активности процесса +

    Mean_InActive_Time_Process = 62  # Среднее время не активности процесса +
    Max_InActive_Time_Process  = 63  # Максимальное время не активности процесса +
    Min_InActive_Time_Process  = 64  # Минимальное время не активности процесса +
    Std_InActive_Time_Process  = 65  # Стандартное отклонение времени не активности процесса +

    # Колличесво обращений к различным корневым веткам реестра
    Appeal_reg_HKCR    = 85  # Количество обращений в ветке реестра HKCR +
    Appeal_reg_HKCU    = 86  # Количество обращений в ветке реестра HKCU +
    Appeal_reg_HKLM    = 87  # Количество обращений в ветке реестра HKLM +
    Appeal_reg_HKU     = 88  # Количество обращений в ветке реестра HKU +
    Appeal_reg_HKCC    = 89  # Количество обращений в ветке реестра HKCC +

    Count_Unique_Reg_Path = 84  # Количество уникальных путей обращения к ключам реестра +

    # Количество событий UDP и TCP
    Count_TCP_Events        = 98  # Количество TCP событий +
    Count_UDP_Events        = 99  # Количество UDP событий +

    # Статистические характеристики сетевых событий в рамках порции
    Ratio_Connect_on_Disconnect = 100  # Соотношение инициированных и завершенных сетевых соединений +
    Ratio_Send_on_Receive       = 101  # Соотношение количества событий отправки и получения +
    Ratio_Send_on_Accept        = 102  # Соотношение количества событий отправки и принятия +
    Ratio_Receive_on_Accept     = 103  # Соотношение количества событий получения и принятия +

    Count_Send_Length = 104  # Размер отправленной по сети информации +
    Mean_Send_Length  = 105  # Средний размер отправляемой по сети информации в одно событие +
    Max_Send_Length   = 106  # Максимальный размер отправляемой по сети информации в одно событие +
    Min_Send_Length   = 107  # Минимальный размер отправляемой по сети информации в одно событие +
    Std_Send_Length   = 108  # Стандартное отклонение размера отправляемой по сети информации в одно событие +

    Count_Receive_Length = 109  # Размер отправленной по сети информации +
    Mean_Receive_Length  = 110  # Средний размер отправляемой по сети информации в одно событие +
    Max_Receive_Length   = 111  # Максимальный размер отправляемой по сети информации в одно событие +
    Min_Receive_Length   = 112  # Минимальный размер отправляемой по сети информации в одно событие +
    Std_Receive_Length   = 113  # Стандартное отклонение размера отправляемой по сети информации в одно событие +

    Count_Unique_Recipients = 114  # Количество уникальных адресов на которые отправляет данные клиент +
    Count_Unique_Ports_src  = 115  # Количество уникальных портов источника +
    Count_Unique_Ports_dst  = 116  # Количество уникальных портов назначения +
    Ratio_src_on_dst_Ports  = 124  # Соотношение количества уникальных портов источника и назначения +

    # Количество событий процессов различных типов
    Count_Process_Defined       = 117  # Количество событий определения процесса
    Count_Thread_Create         = 118  # Количество событий создания потоков
    Count_Thread_Exit           = 119  # Количество событий завершения потоков
    Count_Load_Image            = 120  # Количество событий загрузки образа
    Count_Thread_Profile        = 121  # Количество событий профилирования потока
    Count_Process_Start         = 122  # Количество событий запуска процесса

    Duration = 123  # Длительность порции событий +


class OperationName(Enum):
    # Количество различных операций с файловой системой
    VolumeDismount             = 20  # +
    VolumeMount                = 21  # +
    QueryOpen                  = 22  # +
    CreateFileMapping          = 23  # +
    CreatePipe                 = 24  # +
    QueryInformationFile       = 25  # +
    SetInformationFile         = 26  # +
    QueryEAFile                = 27  # +
    SetEAFile                  = 28  # +
    FlushBuffersFile           = 29  # +
    QueryVolumeInformation     = 30  # +
    SetVolumeInformation       = 31  # +
    DirectoryControl           = 32  # +
    FileSystemControl          = 33  # +
    DeviceIoControl            = 34  # +
    InternalDeviceIoControl    = 35  # +
    LockUnlockFile             = 36  # +
    CreateMailSlot             = 37  # +
    QuerySecurityFile          = 38  # +
    SetSecurityFile            = 39  # +
    SystemControl              = 40  # +
    DeviceChange               = 41  # +
    QueryFileQuota             = 42  # +
    SetFileQuota               = 43  # +
    PlugAndPlay                = 44  # +

    # Количество различных операций с реестром
    OpenKey                  = 66  # Количество событий открытия ключа реестра +
    CreateKey                = 67  # Количество событий создания ключа реестра +
    CloseKey                 = 68  # Количество событий закрытия ключа реестра +
    QueryKey                 = 69  # Количество запросов ключа реестра +
    SetValue                 = 70  # Количество событий установки значения ключа реестра +
    QueryValue               = 71  # Количество запросов значения ключа реестра +
    EnumValue                = 72  # Количество событий перечисления значений реестра +
    EnumKey                  = 73  # Количество событий перечисления ключей реестра +
    SetInfoKey               = 74  # Количество событий установки информационного ключа +
    DeleteKey                = 75  # Количество событий удаления ключа реестра +
    DeleteValue              = 76  # Количество событий удаления значения ключа реестра ++
    FlushKey                 = 77  # Количество событий установки Flush-ключа реестра +
    LoadKey                  = 78  # Количество событий загрузки ключа реестра +
    UnloadKey                = 79  # Количество событий выгрузки ключа реестра +
    RenameKey                = 80  # Количество событий переименования ключа реестра +
    QueryMultipleValueKey    = 81  # Количество запросов ключа с несколькими значениями +
    SetKeySecurity           = 82  # Количество событий установки ключа безопасности +
    QueryKeySecurity         = 83  # Количество запросов ключа безопасности +

    # Количество различных операций с реестром
    Connect       = 90  # Количество инициированных сетевых соединений +
    Disconnect    = 91  # Количество завершённых сетевых соединений +
    Send          = 92  # Количество событий отправки информации по сети +
    Receive       = 93  # Количество событий получения информации по сети +
    Accept        = 94  # Количество событий принятия информации по сети +
    Reconnect     = 95  # Количество событий переподключения +
    Retransmit    = 96  # Количество событий ретрансляции +
    TCPCopy       = 97  # Количество событий TCPCopy +


Event_Charact = [
    "Date & Time",
    "Process Name",
    "Result",
    "Category",
    "Operation",
    "Event Class",
    "Path",
    "Detail"
]


def CulcCharactsEventsOnWindow(events, user_dir):
    characts = {}
    for ch_name in Events_Charact:
        characts[ch_name] = 0

    for operation_name in OperationName:
        characts[operation_name] = 0

    characts[Events_Charact.Time_Stamp_Start]    = events[0]["Date & Time"]
    characts[Events_Charact.Time_Stamp_End]      = events[-1]["Date & Time"]
    characts[Events_Charact.Process_Name]        = events[-1]["Process Name"]
    characts[Events_Charact.Direction_IP_Port]   = None
    characts[Events_Charact.Count_Events_Batch]  = len(events)
    characts[Events_Charact.Duration]            = events[-1]["Date & Time"] - events[0]["Date & Time"]

    Arr_Unique_Path             = {}
    Arr_Read_Length             = array("f")
    Arr_Write_Length            = array("f")

    Arr_Unique_Reg_Path         = {}
    Arr_Unique_Recipients       = {}
    Arr_Unique_Ports_src        = {}
    Arr_Unique_Ports_dst        = {}

    Arr_Send_Length             = array("f")
    Arr_Receive_Length          = array("f")
    arr_time_active             = array("Q")
    arr_time_inactive           = array("Q")

    start_active = events[0]["Date & Time"]

    # bar = tqdm(total=len(events), desc="Выделение характеристик из событий")
    for i in range(len(events)):
        # bar.update(1)
        try:
            if i > 0:
                timeout_inactive = events[i]["Date & Time"] - events[i-1]["Date & Time"]
                if timeout_inactive >= (HUNDREDS_OF_NANOSECONDS * MIN_SECOND_INACTIVE):
                    arr_time_inactive.append(timeout_inactive)
                    arr_time_active.append(events[i-1]["Date & Time"] - start_active)
                    start_active = events[i]["Date & Time"]

            if events[i]["Result"] == 0:
                characts[Events_Charact.Count_Success_Events] += 1

            if "Read" in events[i]["Category"]:
                if "Meta" in events[i]["Category"]:
                    characts[Events_Charact.Count_event_Read_MetaD] += 1
                else:
                    characts[Events_Charact.Count_event_Read] += 1
            elif "Write" in events[i]["Category"]:
                if "Meta" in events[i]["Category"]:
                    characts[Events_Charact.Count_event_Write_MetaD] += 1
                else:
                    characts[Events_Charact.Count_event_Write] += 1

            for operation_name in OperationName:
                if operation_name.name in events[i]["Operation"]:
                    characts[operation_name] += 1

            if events[i]["Event Class"] == EventClass.Process:
                characts[Events_Charact.Count_Event_Process] += 1

                if "Defined" in events[i]["Operation"]:
                    characts[Events_Charact.Count_Process_Defined] += 1
                elif "Create" in events[i]["Operation"]:
                    characts[Events_Charact.Count_Thread_Create] += 1
                elif "Exit" in events[i]["Operation"]:
                    characts[Events_Charact.Count_Thread_Exit] += 1
                elif "Image" in events[i]["Operation"]:
                    characts[Events_Charact.Count_Load_Image] += 1
                elif "Profile" in events[i]["Operation"]:
                    characts[Events_Charact.Count_Thread_Profile] += 1
                elif "Start" in events[i]["Operation"]:
                    characts[Events_Charact.Count_Process_Start] += 1

            elif events[i]["Event Class"] == EventClass.Registry:
                characts[Events_Charact.Count_Event_Registry] += 1

                if not events[i]["Path"] in Arr_Unique_Reg_Path:
                    Arr_Unique_Reg_Path[events[i]["Path"]] = events[i]["Path"]

                if "HKCR" in events[i]["Path"]:
                    characts[Events_Charact.Appeal_reg_HKCR] += 1
                elif "HKCU" in events[i]["Path"]:
                    characts[Events_Charact.Appeal_reg_HKCU] += 1
                elif "HKLM" in events[i]["Path"]:
                    characts[Events_Charact.Appeal_reg_HKLM] += 1
                elif "HKU" in events[i]["Path"]:
                    characts[Events_Charact.Appeal_reg_HKU] += 1
                elif "HKCC" in events[i]["Path"]:
                    characts[Events_Charact.Appeal_reg_HKCC] += 1

            elif events[i]["Event Class"] == EventClass.File_System:
                characts[Events_Charact.Count_Event_File_System] += 1

                if not events[i]["Path"] in Arr_Unique_Path:
                    Arr_Unique_Path[events[i]["Path"]] = events[i]["Path"]

                if "Read" in events[i]["Operation"]:
                    characts[Events_Charact.Count_ReadFile] += 1
                    if "Length" in events[i]["Detail"]:
                        Arr_Read_Length.append(float(str(events[i]["Detail"]["Length"]).replace(",", "")))
                elif "Create" in events[i]["Operation"]:
                    characts[Events_Charact.Count_CreateFile] += 1
                elif "Write" in events[i]["Operation"]:
                    characts[Events_Charact.Count_WriteFile] += 1
                    if "Length" in events[i]["Detail"]:
                        Arr_Write_Length.append(float(str(events[i]["Detail"]["Length"]).replace(",", "")))
                elif "Close" in events[i]["Operation"]:
                    characts[Events_Charact.Count_CloseFile] += 1

            elif events[i]["Event Class"] == EventClass.Network:
                characts[Events_Charact.Count_Event_Network] += 1

                src_dst = str(events[i]["Path"]).split(" -> ")
                if len(src_dst) == 2:
                    idx_spr_port_src = str(src_dst[0]).rfind(":")
                    if idx_spr_port_src != -1:
                        port_src = src_dst[0][idx_spr_port_src+1:]
                        if not port_src in Arr_Unique_Ports_src:
                            Arr_Unique_Ports_src[port_src] = port_src

                    idx_spr_ip_dst = str(src_dst[1]).rfind(":")
                    if idx_spr_ip_dst != -1:
                        direction_check = False

                        ip_dst_str = src_dst[1][:idx_spr_ip_dst]
                        try:

                            ip_dst = IPv4Address(ip_dst_str)
                            direction_check = True
                        except AddressValueError as err:
                            # logging.exception(err)
                            try:
                                ip_dst = gethostbyname(ip_dst_str)
                                direction_check = True
                            except gaierror as err2:
                                ip_dst = ip_dst_str
                                # logging.exception(err2)

                        port_dst_str = src_dst[1][idx_spr_ip_dst+1:]
                        port_dst = None
                        try:
                            port_dst = int(port_dst_str)
                        except ValueError:
                            try:
                                if "UDP" in events[i]["Operation"]:
                                    port_dst = getservbyname(port_dst_str, "udp")
                                elif "TCP" in events[i]["Operation"]:
                                    port_dst = getservbyname(port_dst_str, "tcp")
                            except OSError:
                                port_dst = port_dst_str

                        if direction_check:
                            direction = str(ip_dst) + ":" + str(port_dst)
                            if characts[Events_Charact.Direction_IP_Port] is None:
                                characts[Events_Charact.Direction_IP_Port] = direction
                            elif not direction in characts[Events_Charact.Direction_IP_Port]:
                                characts[Events_Charact.Direction_IP_Port] += (";" + direction)

                        if not ip_dst in Arr_Unique_Recipients:
                            Arr_Unique_Recipients[ip_dst] = ip_dst
                        if not port_dst in Arr_Unique_Ports_dst:
                            Arr_Unique_Ports_dst[port_dst] = port_dst

                if "UDP" in events[i]["Operation"]:
                    characts[Events_Charact.Count_UDP_Events] += 1
                elif "TCP" in events[i]["Operation"]:
                    characts[Events_Charact.Count_TCP_Events] += 1

                if "Length" in events[i]["Detail"]:
                    if "Send" in events[i]["Operation"]:
                        Arr_Send_Length.append(float(str(events[i]["Detail"]["Length"]).replace(",", "")))
                    elif "Receive" in events[i]["Operation"]:
                        Arr_Receive_Length.append(float(str(events[i]["Detail"]["Length"]).replace(",", "")))

        except:
            continue

    # bar.close()

    if characts[Events_Charact.Count_event_Write] != 0:
        characts[Events_Charact.Ratio_Read_on_Write] = characts[Events_Charact.Count_event_Read] / \
                                                         characts[Events_Charact.Count_event_Write]
    else:
        characts[Events_Charact.Ratio_Read_on_Write] = np.inf

    if characts[Events_Charact.Count_WriteFile] != 0:
        characts[Events_Charact.Ratio_Read_on_Write_File] = characts[Events_Charact.Count_ReadFile] / \
                                                            characts[Events_Charact.Count_WriteFile]
    else:
        characts[Events_Charact.Ratio_Read_on_Write_File] = np.inf

    if characts[OperationName.Disconnect] != 0:
        characts[Events_Charact.Ratio_Connect_on_Disconnect] = characts[OperationName.Connect] / \
                                                               characts[OperationName.Disconnect]
    else:
        characts[Events_Charact.Ratio_Connect_on_Disconnect] = np.inf

    if characts[OperationName.Receive] != 0:
        characts[Events_Charact.Ratio_Send_on_Receive] = characts[OperationName.Send] / \
                                                         characts[OperationName.Receive]
    else:
        characts[Events_Charact.Ratio_Send_on_Receive] = np.inf

    if characts[OperationName.Accept] != 0:
        characts[Events_Charact.Ratio_Send_on_Accept] = characts[OperationName.Send] / \
                                                        characts[OperationName.Accept]
    else:
        characts[Events_Charact.Ratio_Send_on_Accept] = np.inf

    if characts[OperationName.Accept] != 0:
        characts[Events_Charact.Ratio_Receive_on_Accept] = characts[OperationName.Receive] / \
                                                           characts[OperationName.Accept]
    else:
        characts[Events_Charact.Ratio_Receive_on_Accept] = np.inf

    characts[Events_Charact.Count_Unique_Path]  = len(Arr_Unique_Path)

    if characts[Events_Charact.Duration] != 0:
        characts[Events_Charact.Speed_Read_Data]  = characts[Events_Charact.Count_Read_Length] / \
                                                    characts[Events_Charact.Duration]
        characts[Events_Charact.Speed_Write_Data]  = characts[Events_Charact.Count_Write_Length] / \
                                                    characts[Events_Charact.Duration]
    else:
        characts[Events_Charact.Speed_Read_Data]  = np.inf
        characts[Events_Charact.Speed_Write_Data] = np.inf

    characts[Events_Charact.Count_Unique_Reg_Path]   = len(Arr_Unique_Reg_Path)
    characts[Events_Charact.Count_Unique_Recipients] = len(Arr_Unique_Recipients)
    characts[Events_Charact.Count_Unique_Ports_src]  = len(Arr_Unique_Ports_src)
    characts[Events_Charact.Count_Unique_Ports_dst]  = len(Arr_Unique_Ports_dst)

    if characts[Events_Charact.Count_Unique_Ports_dst] != 0:
        characts[Events_Charact.Ratio_src_on_dst_Ports] = characts[Events_Charact.Count_Unique_Ports_src] / \
                                                          characts[Events_Charact.Count_Unique_Ports_dst]
    else:
        characts[Events_Charact.Ratio_src_on_dst_Ports] = 0

    if len(Arr_Read_Length) == 0:
        characts[Events_Charact.Count_Read_Length] = 0
        characts[Events_Charact.Mean_Read_Length]  = 0
        characts[Events_Charact.Max_Read_Length]   = 0
        characts[Events_Charact.Min_Read_Length]   = 0
        characts[Events_Charact.Std_Read_Length]   = 0
    else:
        characts[Events_Charact.Count_Read_Length] = np.sum(Arr_Read_Length)
        characts[Events_Charact.Mean_Read_Length]  = np.mean(Arr_Read_Length)
        characts[Events_Charact.Max_Read_Length]   = np.max(Arr_Read_Length)
        characts[Events_Charact.Min_Read_Length]   = np.min(Arr_Read_Length)
        characts[Events_Charact.Std_Read_Length]   = np.std(Arr_Read_Length)

    if len(Arr_Write_Length) == 0:
        characts[Events_Charact.Count_Write_Length] = 0
        characts[Events_Charact.Mean_Write_Length]  = 0
        characts[Events_Charact.Max_Write_Length]   = 0
        characts[Events_Charact.Min_Write_Length]   = 0
        characts[Events_Charact.Std_Write_Length]   = 0
    else:
        characts[Events_Charact.Count_Write_Length] = np.sum(Arr_Write_Length)
        characts[Events_Charact.Mean_Write_Length]  = np.mean(Arr_Write_Length)
        characts[Events_Charact.Max_Write_Length]   = np.max(Arr_Write_Length)
        characts[Events_Charact.Min_Write_Length]   = np.min(Arr_Write_Length)
        characts[Events_Charact.Std_Write_Length]   = np.std(Arr_Write_Length)

    if len(Arr_Send_Length) == 0:
        characts[Events_Charact.Count_Send_Length] = 0
        characts[Events_Charact.Mean_Send_Length]  = 0
        characts[Events_Charact.Max_Send_Length]   = 0
        characts[Events_Charact.Min_Send_Length]   = 0
        characts[Events_Charact.Std_Send_Length]   = 0
    else:
        characts[Events_Charact.Count_Send_Length] = np.sum(Arr_Send_Length)
        characts[Events_Charact.Mean_Send_Length]  = np.mean(Arr_Send_Length)
        characts[Events_Charact.Max_Send_Length]   = np.max(Arr_Send_Length)
        characts[Events_Charact.Min_Send_Length]   = np.min(Arr_Send_Length)
        characts[Events_Charact.Std_Send_Length]   = np.std(Arr_Send_Length)

    if len(Arr_Receive_Length) == 0:
        characts[Events_Charact.Count_Receive_Length] = 0
        characts[Events_Charact.Mean_Receive_Length]  = 0
        characts[Events_Charact.Max_Receive_Length]   = 0
        characts[Events_Charact.Min_Receive_Length]   = 0
        characts[Events_Charact.Std_Receive_Length]   = 0
    else:
        characts[Events_Charact.Count_Receive_Length] = np.sum(Arr_Receive_Length)
        characts[Events_Charact.Mean_Receive_Length]  = np.mean(Arr_Receive_Length)
        characts[Events_Charact.Max_Receive_Length]   = np.max(Arr_Receive_Length)
        characts[Events_Charact.Min_Receive_Length]   = np.min(Arr_Receive_Length)
        characts[Events_Charact.Std_Receive_Length]   = np.std(Arr_Receive_Length)

    if len(arr_time_active) == 0:
        characts[Events_Charact.Mean_Active_Time_Process] = 0
        characts[Events_Charact.Min_Active_Time_Process]  = 0
        characts[Events_Charact.Max_Active_Time_Process]  = 0
        characts[Events_Charact.Std_Active_Time_Process]  = 0
    else:
        characts[Events_Charact.Mean_Active_Time_Process] = np.mean(arr_time_active)
        characts[Events_Charact.Min_Active_Time_Process]  = np.min(arr_time_active)
        characts[Events_Charact.Max_Active_Time_Process]  = np.max(arr_time_active)
        characts[Events_Charact.Std_Active_Time_Process]  = np.std(arr_time_active)

    if len(arr_time_inactive) == 0:
        characts[Events_Charact.Mean_InActive_Time_Process] = 0
        characts[Events_Charact.Min_InActive_Time_Process]  = 0
        characts[Events_Charact.Max_InActive_Time_Process]  = 0
        characts[Events_Charact.Std_InActive_Time_Process]  = 0
    else:
        characts[Events_Charact.Mean_InActive_Time_Process] = np.mean(arr_time_inactive)
        characts[Events_Charact.Min_InActive_Time_Process]  = np.min(arr_time_inactive)
        characts[Events_Charact.Max_InActive_Time_Process]  = np.max(arr_time_inactive)
        characts[Events_Charact.Std_InActive_Time_Process]  = np.std(arr_time_inactive)

    return characts


if __name__ == '__main__':
    from timeit import timeit
    from PMLParser import ParserEvents
    import pandas as pd

    pml_file_name = "F:\\EVENT\\train_events.PML"
    user_dir = "Жертва"
    parser_pml = ParserEvents(pml_file_name, True)
    events = parser_pml.GetEvents()

    timeBool = timeit("""
CulcCharactsEventsOnWindow(events, user_dir)
    """, globals=locals(), number=1)
    print(timeBool)

    characts = CulcCharactsEventsOnWindow(events, user_dir)
    print(characts)
    characts = pd.DataFrame([characts])
    print(characts)