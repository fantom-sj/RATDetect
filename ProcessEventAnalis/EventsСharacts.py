"""
    Модуль содержащий полный набор характеристик событий процессов в системе,
    а также набор характеристик каждого события, необходимых для
    расчёта набора CHARACTERISTIC_EVENTS
"""
from ProcmonParser.consts import EventClass
from statistics import mean, pstdev


CHARACTERISTIC_EVENTS = [
    # Общие сведенья о порции событий
    "Time_Stamp_Start",      # 0. Временная метка начала порции
    "Time_Stamp_End",        # 1. Временная метка конца порции
    "Process_Name",          # 2. Имя процесса
    "Count_Events_Batch",    # 3. Количество событий в порции

    "Count_Success_Events",  # 4. Количество успешных событий

    # Количество событий разной класса
    "Count_Event_Process",      # 5. Количество событий типа Process
    "Count_Event_Registry",     # 6. Количество событий типа Registry
    "Count_Event_File_System",  # 7. Количество событий типа File_System
    "Count_Event_Network",      # 8. Количество событий типа Network

    # Данные по категории событий: чтение и запись данных процессом
    "Count_event_Read",         # 9.  Количество событий относяхися к категории чтение
    "Count_event_Read_MetaD",   # 10. Количество событий относяхися к категории чтение метаданных
    "Count_event_Write",        # 11. Количество событий относяхися к категории записи
    "Count_event_Write_MetaD",  # 12. Количество событий относяхися к категории записи метаданных

    # Основные события работы с файловой системой
    "Count_ReadFile",    # 13. Количество событий на чтение фала
    "Count_CreateFile",  # 14. Количество событий на создание фала
    "Count_WriteFile",   # 15. Количество запросов на запись файла
    "Count_CloseFile",   # 16. Количество запросов на закрытие файла

    # Дополнительные события работы с файловой системой
    "Count_VolumeDismount",           # 17.
    "Count_VolumeMount",              # 18.
    "Count_QueryOpen",                # 19.
    "Count_CreateFileMapping",        # 20.
    "Count_CreatePipe",               # 21.
    "Count_QueryInformationFile",     # 22.
    "Count_SetInformationFile",       # 23.
    "Count_QueryEAFile",              # 24.
    "Count_SetEAFile",                # 25.
    "Count_FlushBuffersFile",         # 26.
    "Count_QueryVolumeInformation",   # 27.
    "Count_SetVolumeInformation",     # 28.
    "Count_DirectoryControl",         # 29.
    "Count_FileSystemControl",        # 30.
    "Count_DeviceIoControl",          # 31.
    "Count_InternalDeviceIoControl",  # 32.
    "Count_LockUnlockFile",           # 33.
    "Count_CreateMailSlot",           # 34.
    "Count_QuerySecurityFile",        # 35.
    "Count_SetSecurityFile",          # 36.
    "Count_SystemControl",            # 37.
    "Count_DeviceChange",             # 38.
    "Count_QueryFileQuota",           # 39.
    "Count_SetFileQuota",             # 40.
    "Count_PlugAndPlay",              # 41.

    # Количество событий относящихся к файловой системе, и различным подкатегориям взаимодействия с ней
    "Count_FilesystemQueryVolumeInformationOperation",  # 42.
    "Count_FilesystemSetVolumeInformationOperation",    # 43.
    "Count_FilesystemQueryInformationOperation",        # 44.
    "Count_FilesystemSetInformationOperation",          # 45.
    "Count_FilesystemDirectoryControlOperation",        # 46.
    "Count_FilesystemPnpOperation",                     # 47.

    # Статистические характеристики относящиеся к работе с файловой системой
    "Count_Unique_Path",     # 48. Количество уникальных путей обращения к файлам
    "Count_Read_Length",     # 49. Количество считанной с диска информации
    "Count_Write_Length",    # 50. Количество записанной на диск информации
    "Max_Read_Length",       # 51. Максимальный объем считанной с диска информации в рамках одного события
    "Min_Read_Length",       # 52. Минимальный объем считанной с диска информации в рамках одного события
    "Mean_Read_Length",      # 53. Средний объем считанной с диска информации в рамках одного события
    "Std_Dev_Read_Length",   # 54. Стандартное отклонение объема считанной с диска информации в рамках одного события
    "Max_Write_Length",      # 55. Максимальный объем записанной на диск информации в рамках одного события
    "Min_Write_Length",      # 56. Минимальный объем записанной на диск информации в рамках одного события
    "Mean_Write_Length",     # 57. Средний объем записанной на диск информации в рамках одного события
    "Std_Dev_Write_Length",  # 58. Стандартное отклонение объема записанной на диск информации в рамках одного события

    # Колличесво обращений к различным ключевым каталогам системы
    "Appeal_to_system32",      # 59. Количество событий с обращением к каталогу Windows/system32
    "Appeal_to_ProgramData",   # 60. Количество событий с обращением к каталогу ProgramData
    "Appeal_to_ProgramFiles",  # 61. Количество событий с обращением к каталогу ProgramFiles
    "Appeal_to_UserDir",       # 62. Количество событий с обращением к домашнему каталогу текущего пользователя

    # Количество событий различных событий реестра
    "Count_Reg_OpenKey",                # 63. Количество событий открытия ключа реестра
    "Count_Reg_CreateKey",              # 64. Количество событий создания ключа реестра
    "Count_Reg_CloseKey",               # 65. Количество событий закрытия ключа реестра
    "Count_Reg_QueryKey",               # 66. Количество запросов ключа реестра
    "Count_Reg_SetValue",               # 67. Количество событий установки значения ключа реестра
    "Count_Reg_QueryValue",             # 68. Количество запросов значения ключа реестра
    "Count_Reg_EnumValue",              # 69. Количество событий перечисления значений реестра
    "Count_Reg_EnumKey",                # 70. Количество событий перечисления ключей реестра
    "Count_Reg_SetInfoKey",             # 71. Количество событий установки информационного ключа
    "Count_Reg_DeleteKey",              # 72. Количество событий удаления ключа реестра
    "Count_Reg_DeleteValue",            # 73. Количество событий удаления значения ключа реестра
    "Count_Reg_FlushKey",               # 74. Количество событий установки Flush-ключа реестра
    "Count_Reg_LoadKey",                # 75. Количество событий загрузки ключа реестра
    "Count_Reg_UnloadKey",              # 76. Количество событий выгрузки ключа реестра
    "Count_Reg_RenameKey",              # 77. Количество событий переименования ключа реестра
    "Count_Reg_QueryMultipleValueKey",  # 78. Количество запросов ключа с несколькими значениями
    "Count_Reg_SetKeySecurity",         # 79. Количество событий установки ключа безопасности
    "Count_Reg_QueryKeySecurity",       # 80. Количество запросов ключа безопасности
    "Count_Unique_Reg_Path",            # 81. Количество уникальных путей обращения к ключам реестра

    # Колличесво обращений к различным корневым веткам реестра
    "Appeal_reg_HKCR",  # 82. Количество обращений в ветке реестра HKCR
    "Appeal_reg_HKCU",  # 83. Количество обращений в ветке реестра HKCU
    "Appeal_reg_HKLM",  # 84. Количество обращений в ветке реестра HKLM
    "Appeal_reg_HKU",   # 85. Количество обращений в ветке реестра HKU
    "Appeal_reg_HKCC",  # 86. Количество обращений в ветке реестра HKCC

    # Количество сетевых событий различных типов
    "Count_Net_Connect",     # 87.  Количество инициированных сетевых соединений
    "Count_Net_Disconnect",  # 88.  Количество завершённых сетевых соединений
    "Count_Net_Send",        # 89.  Количество событий отправки информации по сети
    "Count_Net_Receive",     # 90.  Количество событий получения информации по сети
    "Count_Net_Accept",      # 91.  Количество событий принятия информации по сети
    "Count_Net_Reconnect",   # 92.  Количество событий переподключения
    "Count_Net_Retransmit",  # 93.  Количество событий ретрансляции
    "Count_Net_TCPCopy",     # 94.  Количество событий TCPCopy
    "Count_TCP_Events",      # 95.  Количество TCP событий
    "Count_UDP_Events",      # 96.  Количество UDP событий

    # Статистические характеристики сетевых событий в рамках порции
    "Count_Send_Length",          # 97.   Размер отправленной по сети информации
    "Max_Send_Length",            # 98.   Максимальный размер отправляемой по сети информации в одно событие
    "Min_Send_Length",            # 99.   Минимальный размер отправляемой по сети информации в одно событие
    "Mean_Send_Length",           # 100.  Средний размер отправляемой по сети информации в одно событие
    "Std_Dev_Send_Length",        # 101.  Стандартное отклонение размера отправляемой по сети информации в одно событие
    "Count_Receive_Length",       # 102.  Размер отправленной по сети информации
    "Max_Receive_Length",         # 103.  Максимальный размер отправляемой по сети информации в одно событие
    "Min_Receive_Length",         # 104.  Минимальный размер отправляемой по сети информации в одно событие
    "Mean_Receive_Length",        # 105.  Средний размер отправляемой по сети информации в одно событие
    "Std_Dev_Receive_Length",     # 106.  Стандартное отклонение размера отправляемой по сети информации в одно событие
    "Count_Unique_Recipients",    # 107.  Количество уникальных адресов на которые отправляет данные клиент
    "Count_Unique_Ports_src",     # 108.  Количество уникальных портов источника
    "Count_Unique_Ports_dst",     # 109.  Количество уникальных портов назначения

    # Количество событий процессов различных типов
    "Count_Process_Defined",     # 110. Количество событий определения процесса
    "Count_Thread_Create",       # 111. Количество событий создания потоков
    "Count_Thread_Exit",         # 112. Количество событий завершения потоков
    "Count_Load_Image",          # 113. Количество событий загрузки образа
    "Count_Thread_Profile",      # 114. Количество событий профилирования потока
    "Count_Process_Start",       # 115. Количество событий запуска процесса

    # Различные соотношения
    # "Ratio_Files_on_Reg_Events",  # 9.  Соотношение количества событий файловой системы по отношению к реестру
    # "Ratio_Files_on_Net_Events",  # 10. Соотношение количества событий файловой системы по отношению к сети
    # "Ratio_Reg_on_Net_Events",  # 11. Соотношение количества событий реестра по отношению к сети
    # "Ratio_Read_on_Write",  # 16. Соотношение событий чтения к событиям записи данных
    # "Ratio_Read_on_Write_MD",  # 17. Соотношение событий чтения к событиям записи мета данных
    # "Ratio_TCP_on_UDP",      # 103. Соотношение количества TCP событий к UDP
    # "Ratio_Unq_prt_src_on_dst",  # 117.  Соотношение уникальных портов источника к портам назначения
]

FilesystemQueryVolumeInformationOperation = [
    "QueryInformationVolume",
    "QueryLabelInformationVolume",
    "QuerySizeInformationVolume",
    "QueryDeviceInformationVolume",
    "QueryAttributeInformationVolume",
    "QueryControlInformationVolume",
    "QueryFullSizeInformationVolume",
    "QueryObjectIdInformationVolume"
]

FilesystemSetVolumeInformationOperation = [
    "SetControlInformationVolume",
    "SetLabelInformationVolume",
    "SetObjectIdInformationVolume"
]

FilesystemQueryInformationOperation = [
    "QueryBasicInformationFile",
    "QueryStandardInformationFile",
    "QueryFileInternalInformationFile",
    "QueryEaInformationFile",
    "QueryNameInformationFile",
    "QueryPositionInformationFile",
    "QueryAllInformationFile",
    "QueryEndOfFile",
    "QueryStreamInformationFile",
    "QueryCompressionInformationFile",
    "QueryId",
    "QueryMoveClusterInformationFile",
    "QueryNetworkOpenInformationFile",
    "QueryAttributeTagFile",
    "QueryIdBothDirectory",
    "QueryValidDataLength",
    "QueryShortNameInformationFile",
    "QueryIoPiorityHint",
    "QueryLinks",
    "QueryNormalizedNameInformationFile",
    "QueryNetworkPhysicalNameInformationFile",
    "QueryIdGlobalTxDirectoryInformation",
    "QueryIsRemoteDeviceInformation",
    "QueryAttributeCacheInformation",
    "QueryNumaNodeInformation",
    "QueryStandardLinkInformation",
    "QueryRemoteProtocolInformation",
    "QueryRenameInformationBypassAccessCheck",
    "QueryLinkInformationBypassAccessCheck",
    "QueryVolumeNameInformation",
    "QueryIdInformation",
    "QueryIdExtdDirectoryInformation",
    "QueryHardLinkFullIdInformation",
    "QueryIdExtdBothDirectoryInformation",
    "QueryDesiredStorageClassInformation",
    "QueryStatInformation",
    "QueryMemoryPartitionInformation",
    "QuerySatLxInformation",
    "QueryCaseSensitiveInformation",
    "QueryLinkInformationEx",
    "QueryLinkInfomraitonBypassAccessCheck",
    "QueryStorageReservedIdInformation",
    "QueryCaseSensitiveInformationForceAccessCheck"
]

FilesystemSetInformationOperation = [
    "SetBasicInformationFile",
    "SetRenameInformationFile",
    "SetLinkInformationFile",
    "SetDispositionInformationFile",
    "SetPositionInformationFile",
    "SetAllocationInformationFile",
    "SetEndOfFileInformationFile",
    "SetFileStreamInformation",
    "SetPipeInformation",
    "SetValidDataLengthInformationFile",
    "SetShortNameInformation",
    "SetReplaceCompletionInformation",
    "SetDispositionInformationEx",
    "SetRenameInformationEx",
    "SetRenameInformationExBypassAccessCheck",
    "SetStorageReservedIdInformation"
]

FilesystemDirectoryControlOperation = [
    "QueryDirectory",
    "NotifyChangeDirectory"
]

FilesystemPnpOperation = [
    "StartDevice",
    "QueryRemoveDevice",
    "RemoveDevice",
    "CancelRemoveDevice",
    "StopDevice",
    "QueryStopDevice",
    "CancelStopDevice",
    "QueryDeviceRelations",
    "QueryInterface",
    "QueryCapabilities",
    "QueryResources",
    "QueryResourceRequirements",
    "QueryDeviceText",
    "FilterResourceRequirements",
    "ReadConfig",
    "WriteConfig",
    "Eject",
    "SetLock",
    "QueryId2",
    "QueryPnpDeviceState",
    "QueryBusInformation",
    "DeviceUsageNotification",
    "SurpriseRemoval",
    "QueryLegacyBusInformation"
]


def CulcCharactsEventsOnWindow(events, user_dir):
    characts = {
        # Общие сведенья о порции событий
        "Time_Stamp_Start":     events[0]["Date & Time"],
        "Time_Stamp_End":       events[-1]["Date & Time"],
        "Process_Name":         events[-1]["Process Name"],
        "Count_Events_Batch":   len(events),
        "Count_Success_Events": 0,      #+

        # Количество событий разной класса
        "Count_Event_Process":     0,   #+
        "Count_Event_Registry":    0,   #+
        "Count_Event_File_System": 0,   #+
        "Count_Event_Network":     0,   #+

        # Данные по категории событий: чтение и запись данных процессом
        "Count_event_Read":        0,   #+
        "Count_event_Read_MetaD":  0,   #+
        "Count_event_Write":       0,   #+
        "Count_event_Write_MetaD": 0,   #+

        # Основные события работы с файловой системой
        "Count_ReadFile":   0,   #+
        "Count_CreateFile": 0,   #+
        "Count_WriteFile":  0,   #+
        "Count_CloseFile":  0,   #+

        # Дополнительные события работы с файловой системой
        "Count_VolumeDismount":          0,   #+
        "Count_VolumeMount":             0,   #+
        "Count_QueryOpen":               0,   #+
        "Count_CreateFileMapping":       0,   #+
        "Count_CreatePipe":              0,   #+
        "Count_QueryInformationFile":    0,   #+
        "Count_SetInformationFile":      0,   #+
        "Count_QueryEAFile":             0,   #+
        "Count_SetEAFile":               0,   #+
        "Count_FlushBuffersFile":        0,   #+
        "Count_QueryVolumeInformation":  0,   #+
        "Count_SetVolumeInformation":    0,   #+
        "Count_DirectoryControl":        0,   #+
        "Count_FileSystemControl":       0,   #+
        "Count_DeviceIoControl":         0,   #+
        "Count_InternalDeviceIoControl": 0,   #+
        "Count_LockUnlockFile":          0,   #+
        "Count_CreateMailSlot":          0,   #+
        "Count_QuerySecurityFile":       0,   #+
        "Count_SetSecurityFile":         0,   #+
        "Count_SystemControl":           0,   #+
        "Count_DeviceChange":            0,   #+
        "Count_QueryFileQuota":          0,   #+
        "Count_SetFileQuota":            0,   #+
        "Count_PlugAndPlay":             0,   #+

        # Количество событий относящихся к файловой системе, и различным подкатегориям взаимодействия с ней
        "Count_FilesystemQueryVolumeInformationOperation":  0,   #+
        "Count_FilesystemSetVolumeInformationOperation":    0,   #+
        "Count_FilesystemQueryInformationOperation":        0,   #+
        "Count_FilesystemSetInformationOperation":          0,   #+
        "Count_FilesystemDirectoryControlOperation":        0,   #+
        "Count_FilesystemPnpOperation":                     0,   #+

        # Статистические характеристики относящиеся к работе с файловой системой
        "Count_Unique_Path":    0,   #+
        "Count_Read_Length":    0,   #+
        "Count_Write_Length":   0,   #+
        "Max_Read_Length":      0,   #+
        "Min_Read_Length":      0,   #+
        "Mean_Read_Length":     0,   #+
        "Std_Dev_Read_Length":  0,   #+
        "Max_Write_Length":     0,   #+
        "Min_Write_Length":     0,   #+
        "Mean_Write_Length":    0,   #+
        "Std_Dev_Write_Length": 0,   #+

        # Колличесво обращений к различным ключевым каталогам системы
        "Appeal_to_system32":     0,   #+
        "Appeal_to_ProgramData":  0,   #+
        "Appeal_to_ProgramFiles": 0,   #+
        "Appeal_to_UserDir":      0,   #+

        # Количество событий различных событий реестра
        "Count_Reg_OpenKey":                  0,   #+
        "Count_Reg_CreateKey":                0,   #+
        "Count_Reg_CloseKey":                 0,   #+
        "Count_Reg_QueryKey":                 0,   #+
        "Count_Reg_SetValue":                 0,   #+
        "Count_Reg_QueryValue":               0,   #+
        "Count_Reg_EnumValue":                0,   #+
        "Count_Reg_EnumKey":                  0,   #+
        "Count_Reg_SetInfoKey":               0,   #+
        "Count_Reg_DeleteKey":                0,   #+
        "Count_Reg_DeleteValue":              0,   #+
        "Count_Reg_FlushKey":                 0,   #+
        "Count_Reg_LoadKey":                  0,   #+
        "Count_Reg_UnloadKey":                0,   #+
        "Count_Reg_RenameKey":                0,   #+
        "Count_Reg_QueryMultipleValueKey":    0,   #+
        "Count_Reg_SetKeySecurity":           0,   #+
        "Count_Reg_QueryKeySecurity":         0,   #+
        "Count_Unique_Reg_Path":              0,   #+

        # Колличесво обращений к различным корневым веткам реестра
        "Appeal_reg_HKCR": 0,   #+
        "Appeal_reg_HKCU": 0,   #+
        "Appeal_reg_HKLM": 0,   #+
        "Appeal_reg_HKU":  0,   #+
        "Appeal_reg_HKCC": 0,   #+

        # Количество сетевых событий различных типов
        "Count_Net_Connect":    0,   #+
        "Count_Net_Disconnect": 0,   #+
        "Count_Net_Send":       0,   #+
        "Count_Net_Receive":    0,   #+
        "Count_Net_Accept":     0,   #+
        "Count_Net_Reconnect":  0,   #+
        "Count_Net_Retransmit": 0,   #+
        "Count_Net_TCPCopy":    0,   #+
        "Count_TCP_Events":     0,   #+
        "Count_UDP_Events":     0,   #+

        # Статистические характеристики сетевых событий в рамках порции
        "Count_Send_Length":        0,   #+
        "Max_Send_Length":          0,   #+
        "Min_Send_Length":          0,   #+
        "Mean_Send_Length":         0,   #+
        "Std_Dev_Send_Length":      0,   #+
        "Count_Receive_Length":     0,   #+
        "Max_Receive_Length":       0,   #+
        "Min_Receive_Length":       0,   #+
        "Mean_Receive_Length":      0,   #+
        "Std_Dev_Receive_Length":   0,   #+
        "Count_Unique_Recipients":  0,   #+
        "Count_Unique_Ports_src":   0,   #+
        "Count_Unique_Ports_dst":   0,   #+

        # Количество событий процессов различных типов
        "Count_Process_Defined":   0,   #+
        "Count_Thread_Create":     0,   #+
        "Count_Thread_Exit":       0,   #+
        "Count_Load_Image":        0,   #+
        "Count_Thread_Profile":    0,   #+
        "Count_Process_Start":     0,   #+
        "Count_System_Statistics": 0    #+
    }

    Arr_Unique_Path             = []
    Arr_Read_Length             = []
    Arr_Write_Length            = []

    Arr_Unique_Reg_Path         = []
    Arr_Unique_Recipients       = []
    Arr_Unique_Ports_src        = []
    Arr_Unique_Ports_dst        = []

    Arr_Send_Length             = []
    Arr_Receive_Length          = []

    for i in range(len(events)):
        if events[i]["Result"] == 0:
            characts["Count_Success_Events"] += 1

        if "Read" in events[i]["Category"]:
            if "Meta" in events[i]["Category"]:
                characts["Count_event_Read_MetaD"] += 1
            else:
                characts["Count_event_Read"] += 1
        elif "Write" in events[i]["Category"]:
            if "Meta" in events[i]["Category"]:
                characts["Count_event_Write_MetaD"] += 1
            else:
                characts["Count_event_Write"] += 1

        if events[i]["Event Class"] == EventClass.Process:
            characts["Count_Event_Process"] += 1

            if "Defined" in events[i]["Operation"]:
                characts["Count_Process_Defined"] += 1
            elif "Create" in events[i]["Operation"]:
                characts["Count_Thread_Create"] += 1
            elif "Exit" in events[i]["Operation"]:
                characts["Count_Thread_Exit"] += 1
            elif "Image" in events[i]["Operation"]:
                characts["Count_Load_Image"] += 1
            elif "Profile" in events[i]["Operation"]:
                characts["Count_Thread_Profile"] += 1
            elif "Start" in events[i]["Operation"]:
                characts["Count_Process_Start"] += 1
            elif "Statistics" in events[i]["Operation"]:
                characts["Count_System_Statistics"] += 1

        elif events[i]["Event Class"] == EventClass.Registry:
            characts["Count_Event_Registry"] += 1

            if not events[i]["Path"] in Arr_Unique_Reg_Path:
                Arr_Unique_Reg_Path.append(events[i]["Path"])

            for ch in CHARACTERISTIC_EVENTS[69:87]:
                if ch[10:] in events[i]["Operation"]:
                    characts[ch] += 1

            if "HKCR" in events[i]["Path"]:
                characts["Appeal_reg_HKCR"] += 1
            elif "HKCU" in events[i]["Path"]:
                characts["Appeal_reg_HKCU"] += 1
            elif "HKLM" in events[i]["Path"]:
                characts["Appeal_reg_HKLM"] += 1
            elif "HKU" in events[i]["Path"]:
                characts["Appeal_reg_HKU"] += 1
            elif "HKCC" in events[i]["Path"]:
                characts["Appeal_reg_HKCC"] += 1

        elif events[i]["Event Class"] == EventClass.File_System:
            characts["Count_Event_File_System"] += 1

            if not events[i]["Path"] in Arr_Unique_Path:
                Arr_Unique_Path.append(events[i]["Path"])

            if "Read" in events[i]["Operation"]:
                characts["Count_ReadFile"] += 1
                if "Length" in events[i]["Detail"]:
                    Arr_Read_Length.append(float(str(events[i]["Detail"]["Length"]).replace(",", "")))
            elif "Create" in events[i]["Operation"]:
                characts["Count_CreateFile"] += 1
            elif "Write" in events[i]["Operation"]:
                characts["Count_WriteFile"] += 1
                if "Length" in events[i]["Detail"]:
                    Arr_Write_Length.append(float(str(events[i]["Detail"]["Length"]).replace(",", "")))
            elif "Close" in events[i]["Operation"]:
                characts["Count_CloseFile"] += 1

            for ch in CHARACTERISTIC_EVENTS[22:48]:
                if ch[6:] in events[i]["Operation"]:
                    characts[ch] += 1

            if events[i]["Operation"] in FilesystemQueryVolumeInformationOperation:
                characts["Count_FilesystemQueryVolumeInformationOperation"] += 1
            elif events[i]["Operation"] in FilesystemSetVolumeInformationOperation:
                characts["Count_FilesystemSetVolumeInformationOperation"] += 1
            elif events[i]["Operation"] in FilesystemQueryInformationOperation:
                characts["Count_FilesystemQueryInformationOperation"] += 1
            elif events[i]["Operation"] in FilesystemSetInformationOperation:
                characts["Count_FilesystemSetInformationOperation"] += 1
            elif events[i]["Operation"] in FilesystemDirectoryControlOperation:
                characts["Count_FilesystemDirectoryControlOperation"] += 1
            elif events[i]["Operation"] in FilesystemPnpOperation:
                characts["Count_FilesystemPnpOperation"] += 1

            if "Windows\\System32" in events[i]["Path"]:
                characts["Appeal_to_system32"] += 1
            elif "ProgramData" in events[i]["Path"]:
                characts["Appeal_to_ProgramData"] += 1
            elif "Program Files" in events[i]["Path"]:
                characts["Appeal_to_ProgramFiles"] += 1
            elif user_dir in events[i]["Path"]:
                characts["Appeal_to_UserDir"] += 1

        elif events[i]["Event Class"] == EventClass.Network:
            characts["Count_Event_Network"] += 1

            src_dst = str(events[i]["Path"]).split(" -> ")
            if len(src_dst) == 2:
                idx_spr_port_src = str(src_dst[0]).rfind(":")
                if idx_spr_port_src != -1:
                    port_src = src_dst[0][idx_spr_port_src+1:]
                    if not port_src in Arr_Unique_Ports_src:
                        Arr_Unique_Ports_src.append(port_src)

                idx_spr_ip_dst = str(src_dst[1]).rfind(":")
                if idx_spr_ip_dst != -1:
                    ip_dst   = src_dst[1][:idx_spr_port_src]
                    port_dst = src_dst[1][idx_spr_port_src+1:]
                    if not ip_dst in Arr_Unique_Recipients:
                        Arr_Unique_Recipients.append(ip_dst)
                    if not port_dst in Arr_Unique_Ports_dst:
                        Arr_Unique_Ports_dst.append(port_dst)

            for ch in CHARACTERISTIC_EVENTS[93:101]:
                if ch[10:] in events[i]["Operation"]:
                    characts[ch] += 1

            if "TCP" in events[i]["Operation"]:
                characts["Count_TCP_Events"] += 1
            elif "UDP" in events[i]["Operation"]:
                characts["Count_UDP_Events"] += 1

            if "Length" in events[i]["Detail"]:
                if "Send" in events[i]["Operation"]:
                    Arr_Send_Length.append(float(str(events[i]["Detail"]["Length"]).replace(",", "")))
                elif "Receive" in events[i]["Operation"]:
                    Arr_Receive_Length.append(float(str(events[i]["Detail"]["Length"]).replace(",", "")))

    characts["Count_Unique_Path"]       = len(Arr_Unique_Path)
    characts["Count_Unique_Reg_Path"]   = len(Arr_Unique_Reg_Path)
    characts["Count_Unique_Recipients"] = len(Arr_Unique_Recipients)
    characts["Count_Unique_Ports_src"]  = len(Arr_Unique_Ports_src)
    characts["Count_Unique_Ports_dst"]  = len(Arr_Unique_Ports_dst)

    characts["Max_Read_Length"]         = max(Arr_Read_Length) if len(Arr_Read_Length) > 0 else 0
    characts["Min_Read_Length"]         = min(Arr_Read_Length) if len(Arr_Read_Length) > 0 else 0
    characts["Mean_Read_Length"]        = mean(Arr_Read_Length) if len(Arr_Read_Length) > 0 else 0
    characts["Std_Dev_Read_Length"]     = pstdev(Arr_Read_Length) if len(Arr_Read_Length) > 0 else 0
    characts["Max_Write_Length"]        = max(Arr_Write_Length) if len(Arr_Write_Length) > 0 else 0
    characts["Min_Write_Length"]        = min(Arr_Write_Length) if len(Arr_Write_Length) > 0 else 0
    characts["Mean_Write_Length"]       = mean(Arr_Write_Length) if len(Arr_Write_Length) > 0 else 0
    characts["Std_Dev_Write_Length"]    = pstdev(Arr_Write_Length) if len(Arr_Write_Length) > 0 else 0

    characts["Count_Send_Length"]       = sum(Arr_Send_Length) if len(Arr_Send_Length) > 0 else 0
    characts["Max_Send_Length"]         = max(Arr_Send_Length) if len(Arr_Send_Length) > 0 else 0
    characts["Min_Send_Length"]         = min(Arr_Send_Length) if len(Arr_Send_Length) > 0 else 0
    characts["Mean_Send_Length"]        = mean(Arr_Send_Length) if len(Arr_Send_Length) > 0 else 0
    characts["Std_Dev_Send_Length"]     = pstdev(Arr_Send_Length) if len(Arr_Send_Length) > 0 else 0
    characts["Count_Receive_Length"]    = sum(Arr_Receive_Length) if len(Arr_Receive_Length) > 0 else 0
    characts["Max_Receive_Length"]      = max(Arr_Receive_Length) if len(Arr_Receive_Length) > 0 else 0
    characts["Min_Receive_Length"]      = min(Arr_Receive_Length) if len(Arr_Receive_Length) > 0 else 0
    characts["Mean_Receive_Length"]     = mean(Arr_Receive_Length) if len(Arr_Receive_Length) > 0 else 0
    characts["Std_Dev_Receive_Length"]  = pstdev(Arr_Receive_Length) if len(Arr_Receive_Length) > 0 else 0

    return characts


if __name__ == '__main__':
    print(len(CHARACTERISTIC_EVENTS) - 4)