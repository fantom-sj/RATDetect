"""
    Модуль содержащий полный набор характеристик событий процессов в системе,
    а также набор характеристик каждого события, необходимых для
    расчёта набора CHARACTERISTIC_EVENTS
"""
from pathlib import Path
from ProcmonParser.consts import EventClass


CHARACTERISTIC_EVENTS = [
    "Time_Stamp",               # 0.  Временная метка пакета, с приходом которого были рассчитаны характеристики
    "Process_name",             # 1.  Имя процесса, описываемого набором заданных характеристик

    "Count_Event_Process",      # 2.  Количество событий типа Process
    "Count_Event_Registry",     # 3.  Количество событий типа Registry
    "Count_Event_File_System",  # 4.  Количество событий типа File_System
    "Count_Event_Profiling",    # 5.  Количество событий типа Profiling
    "Count_Event_Network",      # 6.  Количество событий типа Network

    "Count_Read_File",          # 7.  Количество событий на чтение фала
    "Count_Create_File",        # 8.  Количество событий на создание фала
    "Count_Write_File",         # 9.  Количество запросов на запись файла
    "Count_Close_File",         # 10. Количество запросов на закрытие файла
    "Count_QNOIF",              # 11. Количество запросов информации для открытия файла в сети
    "Count_Unique_Path",        # 12. Количество уникальных путей обращения к файлам
    "Count_Read_Length",        # 13. Количество считанной с диска информации
    "Count_Write_Length",       # 14. Количество записанной на диск информации

    "Count_Reg_Open_Key",       # 15. Количество событий открытия ключа реестра
    "Count_Reg_Create_Key",     # 16. Количество событий создания ключа реестра
    "Count_Reg_Close_Key",      # 17. Количество событий закрытия ключа реестра
    "Count_Reg_Query_Key",      # 18. Количество запросов ключа реестра
    "Count_Reg_Set_Value",      # 19. Количество событий установки значения ключа реестра
    "Count_Reg_Query_Value",    # 20. Количество запроса значения ключа реестра
    "Count_Reg_Delete_Key",     # 21. Количество событий удаления ключа реестра
    "Count_Reg_Delete_Value",   # 22. Количество событий удаления значения ключа реестра
    "Count_Reg_Enum_Value",     # 23. Количество событий перечисления значений ключей реестра
    "Count_Reg_Enum_Key",       # 24. Количество событий перечисления ключей реестра
    "Count_Reg_Other_Event",    # 25. Количество других событий реестра
    "Count_Unique_Reg_Path",    # 26. Количество уникальных путей обращения к файлам

    "Count_Net_Connect",        # 27. Количество инициированных сетевых соединений
    "Count_Net_Disconnect",     # 28. Количество завершённых сетевых соединений
    "Count_Net_Send",           # 29. Количество сетевых отправлений
    "Count_Net_Receive",        # 30. Количество сетевых приёмов
    "Count_Send_Length",        # 31. Количество отправленной по сети информации
    "Count_Receive_Length",     # 32. Количество принятой по сети информации
    "Count_Unique_Recipients",  # 33. Количество уникальных адресатов

    "Count_Create_Thread",      # 34. Количество созданных потоков
    "Count_Exit_Thread",        # 35. Количество завершённых потоков

    # "Count_event_Unknown",      # 6.  Количество событий типа Unknown
    # "Count_success_events",     # 7.  Количество успешно завершённых событий в заданном промежутке

    # "Access_to_system32",       # 8.  Количество событий с попыткой доступа к папке Windows/system32
    # "Access_to_ProgramData",    # 9.  Количество событий с попыткой доступа к папке ProgramData
    # "Access_to_ProgramFiles",   # 10. Количество событий с попыткой доступа к папке ProgramFiles
    # "Access_to_UserDir",        # 11. Количество событий с попыткой доступа к домашней папке пользователя
    # "Access_to_other",          # 12. Количество событий с попыткой доступа к остальным папкам


    # "Ratio_TCP_on_UDP",         # 17. Соотношение количества TCP операций к UDP

    # "Count_create_thread",      # 18. Количество созданных потоков
    # "Count_exit_thread",        # 19. Количество выходов из потоков

    # "Access_reg_HKCR",          # 20. Количество обращений в ветке реестра HKCR
    # "Access_reg_HKCU",          # 21. Количество обращений в ветке реестра HKCU
    # "Access_reg_HKLM",          # 22. Количество обращений в ветке реестра HKLM
    # "Access_reg_HKU",           # 23. Количество обращений в ветке реестра HKU
    # "Access_reg_HKCC",          # 24. Количество обращений в ветке реестра HKCC

    # "Ratio_files_on_reg",       # 25. Соотношение количества запросов на доступ к файлам по отношению к реестру
    # "Ratoi_read_on_write_file", # 26. Соотношение количества операция чтения к записи файлов
    # "Ratoi_read_on_write_reg",  # 27. Соотношение количества операция чтения к записи в реестре
    # "Ratoi_read_on_write_data", # 28. Соотношение количества прочитанных данных к записанным данным на диск
]


def CulcCharactsEventsOnWindow(events, window_size):
    Time_Stamp                  = events[-1]["Date & Time"]
    Process_name                = events[-1]["Process Name"]

    Count_Event_Process         = 0
    Count_Event_Registry        = 0
    Count_Event_File_System     = 0
    Count_Event_Profiling       = 0
    Count_Event_Network         = 0

    Count_Read_File             = 0
    Count_Create_File           = 0
    Count_Write_File            = 0
    Count_Close_File            = 0
    Count_QNOIF                 = 0
    Count_Read_Length           = 0
    Count_Write_Length          = 0

    Count_Reg_Open_Key          = 0
    Count_Reg_Create_Key        = 0
    Count_Reg_Close_Key         = 0
    Count_Reg_Query_Key         = 0
    Count_Reg_Set_Value         = 0
    Count_Reg_Query_Value       = 0
    Count_Reg_Delete_Key        = 0
    Count_Reg_Delete_Value      = 0
    Count_Reg_Enum_Value        = 0
    Count_Reg_Enum_Key          = 0
    Count_Reg_Other_Event       = 0

    Count_Net_Connect           = 0
    Count_Net_Disconnect        = 0
    Count_Net_Send              = 0
    Count_Net_Receive           = 0
    Count_Send_Length           = 0
    Count_Receive_Length        = 0

    Count_Create_Thread         = 0
    Count_Exit_Thread           = 0

    Arr_Path_File               = []
    Arr_Path_Reg                = []
    Arr_Unique_Recipients       = []

    for i in range(window_size):
        if events[i]["Event Class"] == EventClass.Process:
            Count_Event_Process += 1

            if "Create" in events[i]["Operation"]:
                Count_Create_Thread += 1
            elif "Exit" in events[i]["Operation"]:
                Count_Exit_Thread += 1

        elif events[i]["Event Class"] == EventClass.Registry:
            Count_Event_Registry += 1

            if not events[i]["Path"] in Arr_Path_Reg:
                Arr_Path_Reg.append(events[i]["Path"])

            if "OpenKey" in events[i]["Operation"]:
                Count_Reg_Open_Key      += 1
            elif "CreateKey" in events[i]["Operation"]:
                Count_Reg_Create_Key    += 1
            elif "CloseKey" in events[i]["Operation"]:
                Count_Reg_Close_Key     += 1
            elif "QueryKey" in events[i]["Operation"]:
                Count_Reg_Query_Key     += 1
            elif "SetValue" in events[i]["Operation"]:
                Count_Reg_Set_Value     += 1
            elif "QueryValue" in events[i]["Operation"]:
                Count_Reg_Query_Value   += 1
            elif "DeleteKey" in events[i]["Operation"]:
                Count_Reg_Delete_Key    += 1
            elif "DeleteValue" in events[i]["Operation"]:
                Count_Reg_Delete_Value  += 1
            elif "EnumValue" in events[i]["Operation"]:
                Count_Reg_Enum_Value    += 1
            elif "EnumKey" in events[i]["Operation"]:
                Count_Reg_Enum_Key      += 1
            else:
                Count_Reg_Other_Event   += 1

        elif events[i]["Event Class"] == EventClass.File_System:
            Count_Event_File_System += 1

            if not events[i]["Path"] in Arr_Path_File:
                Arr_Path_File.append(events[i]["Path"])

            try:
                # Подсчитываем количество операций чтения и записи файлов
                if "Read" in events[i]["Operation"]:
                    Count_Read_File     += 1
                    Count_Read_Length   += events[i]["Detail"]["Length"]
                elif "Write" in events[i]["Operation"]:
                    Count_Write_File    += 1
                    Count_Write_Length  += events[i]["Detail"]["Length"]
                elif "Create" in events[i]["Operation"]:
                    Count_Create_File   += 1
                elif "Close" in events[i]["Operation"]:
                    Count_Close_File    += 1
                elif "QueryNetworkOpenInformationFile" in events[i]["Operation"]:
                    Count_QNOIF         += 1
            except:
                continue

        elif events[i]["Event Class"] == EventClass.Profiling:
            Count_Event_Profiling += 1

        elif events[i]["Event Class"] == EventClass.Network:
            Count_Event_Network += 1

            if not events[i]["Path"] in Arr_Unique_Recipients:
                Arr_Unique_Recipients.append(events[i]["Path"])

            try:
                if "Connect" in events[i]["Operation"]:
                    Count_Net_Connect       += 1
                elif "Disconnect" in events[i]["Operation"]:
                    Count_Net_Disconnect    += 1
                elif "Send" in events[i]["Operation"]:
                    Count_Net_Send          += 1
                    Count_Send_Length       += events[i]["Detail"]["Length"]
                elif "Receive" in events[i]["Operation"]:
                    Count_Net_Receive       += 1
                    Count_Receive_Length    += events[i]["Detail"]["Length"]
            except:
                continue

    Count_Unique_Path       = len(Arr_Path_File)
    Count_Unique_Reg_Path   = len(Arr_Path_Reg)
    Count_Unique_Recipients = len(Arr_Unique_Recipients)

    characts = {
        "Time_Stamp":               Time_Stamp,
        "Process_name":             Process_name,
        "Count_Event_Process":      Count_Event_Process,
        "Count_Event_Registry":     Count_Event_Registry,
        "Count_Event_File_System":  Count_Event_File_System,
        "Count_Event_Profiling":    Count_Event_Profiling,
        "Count_Event_Network":      Count_Event_Network,
        "Count_Read_File":          Count_Read_File,
        "Count_Create_File":        Count_Create_File,
        "Count_Write_File":         Count_Write_File,
        "Count_Close_File":         Count_Close_File,
        "Count_QNOIF":              Count_QNOIF,
        "Count_Unique_Path":        Count_Unique_Path,
        "Count_Read_Length":        Count_Read_Length,
        "Count_Write_Length":       Count_Write_Length,
        "Count_Reg_Open_Key":       Count_Reg_Open_Key,
        "Count_Reg_Create_Key":     Count_Reg_Create_Key,
        "Count_Reg_Close_Key":      Count_Reg_Close_Key,
        "Count_Reg_Query_Key":      Count_Reg_Query_Key,
        "Count_Reg_Set_Value":      Count_Reg_Set_Value,
        "Count_Reg_Query_Value":    Count_Reg_Query_Value,
        "Count_Reg_Delete_Key":     Count_Reg_Delete_Key,
        "Count_Reg_Delete_Value":   Count_Reg_Delete_Value,
        "Count_Reg_Enum_Value":     Count_Reg_Enum_Value,
        "Count_Reg_Enum_Key":       Count_Reg_Enum_Key,
        "Count_Reg_Other_Event":    Count_Reg_Other_Event,
        "Count_Unique_Reg_Path":    Count_Unique_Reg_Path,
        "Count_Net_Connect":        Count_Net_Connect,
        "Count_Net_Disconnect":     Count_Net_Disconnect,
        "Count_Net_Send":           Count_Net_Send,
        "Count_Net_Receive":        Count_Net_Receive,
        "Count_Send_Length":        Count_Send_Length,
        "Count_Receive_Length":     Count_Receive_Length,
        "Count_Unique_Recipients":  Count_Unique_Recipients,
        "Count_Create_Thread":      Count_Create_Thread,
        "Count_Exit_Thread":        Count_Exit_Thread
    }

    return characts


def CulcCharactsEventsOnWindowOLD(events, window_size):
    Time_Stamp              = events[-1]["Date & Time"]
    Process_name            = events[-1]["Process Name"]

    Count_event_Process     = 0
    Count_event_Registry    = 0
    Count_event_File_System = 0
    Count_event_Profiling   = 0
    Count_event_Network     = 0
    Count_event_Unknown     = 0
    Count_success_events    = 0

    Access_to_system32      = 0
    Access_to_ProgramData   = 0
    Access_to_ProgramFiles  = 0
    Access_to_UserDir       = 0
    Access_to_other         = 0

    Count_net_Connect       = 0
    Count_net_Disconnect    = 0
    Count_net_Send          = 0
    Count_net_Receive       = 0
    Count_net_TCP           = 0
    Count_net_UDP           = 0

    Count_create_thread     = 0
    Count_exit_thread       = 0

    Access_reg_HKCR         = 0
    Access_reg_HKCU         = 0
    Access_reg_HKLM         = 0
    Access_reg_HKU          = 0
    Access_reg_HKCC         = 0

    Count_Read_file         = 0
    Count_Write_file        = 0
    Count_Read_reg          = 0
    Count_Write_reg         = 0

    Len_data_read           = 0
    Len_data_write          = 0

    for i in range(window_size):
        # Подсчитываем количество успешных операций
        if events[i]["Result"] == 0:
            Count_success_events += 1

        # Подсчитываем колличество операций каждого типа
        if events[i]["Event Class"] == EventClass.Process:
            Count_event_Process += 1

            # Подсчет количества разных операций с потоками
            if "Create" in events[i]["Operation"]:
                Count_create_thread += 1
            elif "Exit" in events[i]["Operation"]:
                Count_exit_thread += 1

        elif events[i]["Event Class"] == EventClass.Registry:
            Count_event_Registry += 1

            # Подсчет попыток доступа к разным корневым веткам реестра
            if "HKCR" in events[i]["Path"]:
                Access_reg_HKCR += 1
            elif "HKCU" in events[i]["Path"]:
                Access_reg_HKCU += 1
            elif "HKLM" in events[i]["Path"]:
                Access_reg_HKLM += 1
            elif "HKU" in events[i]["Path"]:
                Access_reg_HKU += 1
            elif "HKCC" in events[i]["Path"]:
                Access_reg_HKCC += 1

            # Подсчитываем количество операций чтения и записи файлов
            if "Read" in events[i]["Category"]:
                Count_Read_reg += 1
            elif "Write" in events[i]["Category"]:
                Count_Write_reg += 1

        elif events[i]["Event Class"] == EventClass.File_System:
            Count_event_File_System += 1

            # Подсчитываем количество операций на доступ к различным системным директориям
            if "Windows\\System32" in events[i]["Path"] or "Windows\\system32" in events[i]["Path"]:
                Access_to_system32 += 1
            elif "ProgramData" in events[i]["Path"]:
                Access_to_ProgramData += 1
            elif "Program Files" in events[i]["Path"]:
                Access_to_ProgramFiles += 1
            elif str(Path.home()) in events[i]["Path"]:
                Access_to_UserDir += 1
            else:
                Access_to_other += 1

            try:
                # Подсчитываем количество операций чтения и записи файлов
                if "Read" in events[i]["Category"]:
                    Count_Read_file += 1
                    Len_data_read += events[i]["Detail"]["Length"]
                elif "Write" in events[i]["Category"]:
                    Count_Write_file += 1
                    Len_data_write += events[i]["Detail"]["Length"]
            except:
                continue

        elif events[i]["Event Class"] == EventClass.Profiling:
            Count_event_Profiling += 1

        elif events[i]["Event Class"] == EventClass.Network:
            Count_event_Network += 1

            # Подсчитываем количество сетевых операций разного типа
            if "Connect" in events[i]["Operation"]:
                Count_net_Connect += 1
            elif "Disconnect" in events[i]["Operation"]:
                Count_net_Disconnect += 1
            elif "Send" in events[i]["Operation"]:
                Count_net_Send += 1
            elif "Receive" in events[i]["Operation"]:
                Count_net_Receive += 1

            if "TCP" in events[i]["Operation"]:
                Count_net_TCP += 1
            elif "UDP" in events[i]["Operation"]:
                Count_net_UDP += 1

        elif events[i]["Event Class"] == EventClass.Unknown:
            Count_event_Unknown += 1

    # Соотношение количества TCP операций к UDP
    if Count_net_UDP > 0:
        Ratio_TCP_on_UDP = Count_net_TCP / Count_net_UDP
    else:
        Ratio_TCP_on_UDP = 1

    # Соотношение количества запросов на доступ к файлам по отношению к реестру
    if Count_event_Registry > 0:
        Ratio_files_on_reg = Count_event_File_System / Count_event_Registry
    else:
        Ratio_files_on_reg = 1

    # Соотношение количества операция чтения к записи файлов
    if Count_Write_file > 0:
        Ratoi_read_on_write_file = Count_Read_file / Count_Write_file
    else:
        Ratoi_read_on_write_file = 1

    # Соотношение количества операция чтения к записи в реестре
    if Count_Write_reg > 0:
        Ratoi_read_on_write_reg = Count_Read_reg / Count_Write_reg
    else:
        Ratoi_read_on_write_reg = 1

    # Соотношение количества прочитанных данных к записанным данным на диск
    if Len_data_write > 0:
        Ratoi_read_on_write_data = Len_data_read / Len_data_write
    else:
        Ratoi_read_on_write_data = 1

    characts = {
        "Time_Stamp":               Time_Stamp,
        "Process_name":             Process_name,
        "Count_event_Process":      Count_event_Process,
        "Count_event_Registry":     Count_event_Registry,
        "Count_event_File_System":  Count_event_File_System,
        "Count_event_Profiling":    Count_event_Profiling,
        "Count_event_Network":      Count_event_Network,
        "Count_event_Unknown":      Count_event_Unknown,
        "Count_success_events":     Count_success_events,
        "Access_to_system32":       Access_to_system32,
        "Access_to_ProgramData":    Access_to_ProgramData,
        "Access_to_ProgramFiles":   Access_to_ProgramFiles,
        "Access_to_UserDir":        Access_to_UserDir,
        "Access_to_other":          Access_to_other,
        "Count_net_Connect":        Count_net_Connect,
        "Count_net_Disconnect":     Count_net_Disconnect,
        "Count_net_Send":           Count_net_Send,
        "Count_net_Receive":        Count_net_Receive,
        "Ratio_TCP_on_UDP":         Ratio_TCP_on_UDP,
        "Count_create_thread":      Count_create_thread,
        "Count_exit_thread":        Count_exit_thread,
        "Access_reg_HKCR":          Access_reg_HKCR,
        "Access_reg_HKCU":          Access_reg_HKCU,
        "Access_reg_HKLM":          Access_reg_HKLM,
        "Access_reg_HKU":           Access_reg_HKU,
        "Access_reg_HKCC":          Access_reg_HKCC,
        "Ratio_files_on_reg":       Ratio_files_on_reg,
        "Ratoi_read_on_write_file": Ratoi_read_on_write_file,
        "Ratoi_read_on_write_reg":  Ratoi_read_on_write_reg,
        "Ratoi_read_on_write_data": Ratoi_read_on_write_data
    }

    return characts