"""
    Модуль содержащий полный набор характеристик событий процессов в системе,
    а также набор характеристик каждого события, необходимых для
    расчёта набора CHARACTERISTIC_EVENTS
"""
from pathlib import Path
from ProcmonParser.consts import EventClass


CHARACTERISTIC_EVENTS = [
    "Time_Stamp",               # 0.  Временная метка пакета, с приходом которого были рассчитаны характеристики

    "Count_event_Process",      # 1.  Количество событий типа Process +
    "Count_event_Registry",     # 2.  Количество событий типа Registry +
    "Count_event_File_System",  # 3.  Количество событий типа File_System +
    "Count_event_Profiling",    # 4.  Количество событий типа Profiling +
    "Count_event_Network",      # 5.  Количество событий типа Network +
    "Count_event_Unknown",      # 6.  Количество событий типа Unknown +
    "Count_success_events",     # 7.  Количество успешно завершённых событий в заданном промежутке +

    "Access_to_system32",       # 8.  Количество событий с попыткой доступа к папке Windows/system32 +
    "Access_to_ProgramData",    # 9.  Количество событий с попыткой доступа к папке ProgramData +
    "Access_to_ProgramFiles",   # 10. Количество событий с попыткой доступа к папке ProgramFiles +
    "Access_to_UserDir",        # 11. Количество событий с попыткой доступа к домашней папке пользователя +
    "Access_to_other",          # 12. Количество событий с попыткой доступа к остальным папкам +

    "Count_net_Connect",        # 13. Количество инициированных сетевых соединений +
    "Count_net_Disconnect",     # 14. Количество завершённых сетевых соединений +
    "Count_net_Send",           # 15. Количество сетевых отправлений +
    "Count_net_Receive",        # 16. Количество сетевых приёмов +
    "Ratio_TCP_on_UDP",         # 17. Соотношение количества TCP операций к UDP +

    "Count_create_thread",      # 18. Количество созданных потоков +
    "Count_exit_thread",        # 19. Количество выходов из потоков +

    "Access_reg_HKCR",          # 20. Количество обращений в ветке реестра HKCR +
    "Access_reg_HKCU",          # 21. Количество обращений в ветке реестра HKCU +
    "Access_reg_HKLM",          # 22. Количество обращений в ветке реестра HKLM +
    "Access_reg_HKU",           # 23. Количество обращений в ветке реестра HKU +
    "Access_reg_HKCC",          # 24. Количество обращений в ветке реестра HKCC +

    "Ratio_files_on_reg",       # 25. Соотношение количества запросов на доступ к файлам по отношению к реестру +
    "Ratoi_read_on_write_file", # 26. Соотношение количества операция чтения к записи файлов +
    "Ratoi_read_on_write_reg",  # 27. Соотношение количества операция чтения к записи в реестре +
    "Ratoi_read_on_write_data", # 28. Соотношение количества прочитанных данных к записанным данным на диск
]


def CulcCharactsEventsOnWindow(events, window_size):
    Time_Stamp = events[-1]["Date & Time"]

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
        "Time_Stamp": Time_Stamp,
        "Count_event_Process": Count_event_Process,
        "Count_event_Registry": Count_event_Registry,
        "Count_event_File_System": Count_event_File_System,
        "Count_event_Profiling": Count_event_Profiling,
        "Count_event_Network": Count_event_Network,
        "Count_event_Unknown": Count_event_Unknown,
        "Count_success_events": Count_success_events,
        "Access_to_system32": Access_to_system32,
        "Access_to_ProgramData": Access_to_ProgramData,
        "Access_to_ProgramFiles": Access_to_ProgramFiles,
        "Access_to_UserDir": Access_to_UserDir,
        "Access_to_other": Access_to_other,
        "Count_net_Connect": Count_net_Connect,
        "Count_net_Disconnect": Count_net_Disconnect,
        "Count_net_Send": Count_net_Send,
        "Count_net_Receive": Count_net_Receive,
        "Ratio_TCP_on_UDP": Ratio_TCP_on_UDP,
        "Count_create_thread": Count_create_thread,
        "Count_exit_thread": Count_exit_thread,
        "Access_reg_HKCR": Access_reg_HKCR,
        "Access_reg_HKCU": Access_reg_HKCU,
        "Access_reg_HKLM": Access_reg_HKLM,
        "Access_reg_HKU": Access_reg_HKU,
        "Access_reg_HKCC": Access_reg_HKCC,
        "Ratio_files_on_reg": Ratio_files_on_reg,
        "Ratoi_read_on_write_file": Ratoi_read_on_write_file,
        "Ratoi_read_on_write_reg": Ratoi_read_on_write_reg,
        "Ratoi_read_on_write_data": Ratoi_read_on_write_data
    }

    return characts