from ProcessEventAnalis.EventsСharacts import CulcCharactsEventsOnWindow, Event_Charact, \
    HUNDREDS_OF_NANOSECONDS, Events_Charact
from ProcessEventAnalis.PMLParser import ParserEvents
from threading import Thread
from pathlib import Path
from tqdm import tqdm

import pandas as pd
import logging
import pickle
import socket
import struct
import time
import ssl
import math


class ProcessThread:
    def __init__(self):
        """
            Возможные статусы процесса:
            None - процесс только создан, статус отсутствует
            0 - процесс завершён
            1 - процесс в активном состоянии
            -1 - поток прерван из-за временных рамок
        """
        self.current_thread = None
        self.status       = None
        self.events       = {}

    def append(self, event):
        if "Process" in event["Operation"] and "Exit" in event["Operation"]:
            self.status = 0
            self.events[self.current_thread].append(event)
        else:
            self.events[self.current_thread].append(event)


class ProcessThreads:
    def __init__(self, thread_time_limit):
        self.threads = {}
        self.thread_time_limit = thread_time_limit

    def creatThread(self, event: dict, name_thread):
        if not name_thread in self.threads:
            self.threads[name_thread] = ProcessThread()

        self.threads[name_thread].status = 1
        self.threads[name_thread].current_thread = event["Date & Time"]
        self.threads[name_thread].events[self.threads[name_thread].current_thread] = list()
        self.threads[name_thread].append(event)

    def delThread(self, name_thread, timeThread):
        del self.threads[name_thread].events[timeThread]
        if not self.threads[name_thread].events:
            del self.threads[name_thread]

    def appendEvent(self, event: dict):
        name_thread = event["Process Name"]

        if not name_thread in self.threads:
            self.creatThread(event, name_thread)
        else:
            if self.threads[name_thread].status == 1:
                if event["Date & Time"] - self.threads[name_thread].current_thread >= self.thread_time_limit:
                    self.threads[name_thread].status = -1
                    self.creatThread(event, name_thread)
                else:
                    self.threads[name_thread].append(event)
            else:
                self.creatThread(event, name_thread)

    def printInFile(self, file_name):
        max_subthread = 0
        add_events = 0
        with open(file_name, "w") as printthr:
            for nameThread in self.threads:
                index = 0
                printthr.write(f"{nameThread} = {'{'}\n")
                for timeThread in self.threads[nameThread].events:
                    index += 1
                    if index > max_subthread:
                        max_subthread = index
                    printthr.write(f"\t{timeThread}: {'{'}\n")
                    for idx in range(len(self.threads[nameThread].events[timeThread])):
                        event = self.threads[nameThread].events[timeThread][idx]
                        add_events += 1
                        printthr.write(f"\t\t{idx}: {'{'}\n")

                        for ch in Event_Charact:
                            printthr.write(f"\t\t\t{ch}: {event[ch]},\n")
                        printthr.write(f"\t\t{'},'}\n")
                    printthr.write(f"\t{'},'}\n")
                printthr.write(f"{'}'}\n")

        print(f"Максимальное количество потоков с одинаковым именем: {max_subthread}")
        print(f"Было записано {add_events} пакетов")
        print("Запись завершена!")


class AnalyzerEvents(Thread):
    def __init__(self, thread_time_limit, charact_file_mask, path_name, user_dir, HOST, PORT, SERVER_HOST, SERVER_PORT):
        super().__init__()

        self.thread_time_limit      = thread_time_limit
        self.charact_file_mask      = charact_file_mask
        self.path_name              = path_name
        self.user_dir               = user_dir

        self.files_events_arr       = list()
        self.array_events_global    = list()
        self.process_arr            = list()

        self.ProcessThreads_obj = ProcessThreads(self.thread_time_limit)

        self.run_analyz             = True
        self.th_main_analyz         = None
        self.index_charact_file     = -1

        self.HOST        = HOST
        self.PORT        = PORT
        self.SERVER_HOST = SERVER_HOST
        self.SERVER_PORT = SERVER_PORT

        self.buffer_waiting = pd.DataFrame()
        self.max_len_buffer = 20

        self.context = ssl.SSLContext(ssl.PROTOCOL_TLS_CLIENT)
        self.context.load_verify_locations("ca.crt")

        self.GetFilesEvents()

    def GetFilesEvents(self):
        path_home = Path(self.path_name)
        files_local      = {}
        files_timecreate = []

        for file in path_home.iterdir():
            file_split = str(file).split(".")

            if file.is_file():
                if file_split[1] == "pml" or file_split[1] == "PML":
                    size_file = file.stat().st_size
                    if size_file > 0:
                        try:
                            old_name_file = str(file)
                            new_name_file = old_name_file + "_tmp"
                            file.rename(new_name_file)
                            Path(new_name_file).rename(old_name_file)
                            time_create = file.stat().st_mtime
                            files_timecreate.append(time_create)
                            files_local[time_create] = str(file)
                        except PermissionError:
                            continue
                    else:
                        continue
                else:
                    continue
            else:
                continue

        self.files_events_arr = []
        if len(files_timecreate) > 0:
            files_timecreate.sort()

            for tc in files_timecreate:
                self.files_events_arr.append(files_local[tc])

        return len(self.files_events_arr)

    def GetLastFileId(self):
        path_home = Path(self.path_name)
        file_arr = []

        for file in path_home.iterdir():
            file = str(file)
            if not (".csv" in file):
                continue
            else:
                file_arr.append(file)

        if len(file_arr) > 0:
            indexs_files_events = []

            # Получение индексов файлов с событиями
            for file_name in file_arr:
                index_file = int(file_name.split(".")[0].split("\\")[-1].split("_")[-1])
                indexs_files_events.append(index_file)

            indexs_files_events.sort()

            self.index_charact_file = indexs_files_events[-1]
        else:
            self.index_charact_file = -1
        return self.index_charact_file

    def ParsePMLFile(self, file_name):
        print(f"\nЗапускаем обработку файла: {file_name}")
        pbar = tqdm(desc="Загрузка событий")
        parser = ParserEvents(file_name)
        for event in parser.GenEventIter():
            self.array_events_global.append(event)
            pbar.update(1)
        pbar.close()

        path_new = self.path_name + "\\" + "Обработанные файлы"
        if not Path(path_new).exists():
            Path(path_new).mkdir()
        file_only_name = file_name.split("\\")[-1]
        Path(file_name).rename(path_new + "\\" + file_only_name)

    def ProcessingEvents(self, arr_pml_file=None):
        if not arr_pml_file is None:
            for file in arr_pml_file:
                self.ParsePMLFile(file)

        pbar = tqdm(total=len(self.array_events_global), desc="Сортировка событий")
        for event in self.array_events_global:
            pbar.update(1)
            if "python" in event["Process Name"] or "pycharm" in \
                    event["Process Name"] or "Procmon" in event["Process Name"]:
                continue
            self.ProcessThreads_obj.appendEvent(event)

        self.array_events_global.clear()
        pbar.close()

        array_characts  = list()
        total = len(self.ProcessThreads_obj.threads)
        for nameThread in list(self.ProcessThreads_obj.threads):
            total += len(self.ProcessThreads_obj.threads[nameThread].events)

        pbar = tqdm(total=total, desc="Анализ событий")
        for nameThread in list(self.ProcessThreads_obj.threads):
            if not nameThread in self.ProcessThreads_obj.threads:
                continue

            for timeThread in list(self.ProcessThreads_obj.threads[nameThread].events):
                pbar.update(1)
                if not timeThread in self.ProcessThreads_obj.threads[nameThread].events:
                    continue

                if self.ProcessThreads_obj.threads[nameThread].current_thread == timeThread and \
                        self.ProcessThreads_obj.threads[nameThread].status == 1:
                    continue
                else:
                    thread = self.ProcessThreads_obj.threads[nameThread].events[timeThread]
                    threadCharacts = CulcCharactsEventsOnWindow(thread, self.user_dir)
                    self.ProcessThreads_obj.delThread(nameThread, timeThread)
                    array_characts.append(threadCharacts)

        pbar.close()

        if len(array_characts) == 0:
            print("Не выявлено ни одного набора характеристик!")
            return False
        else:
            try:
                print(f"Выявлены {len(array_characts)} характеристики событий процессов")
                pd_characts = pd.DataFrame(array_characts)
                pd_characts.sort_values(by=Events_Charact.Time_Stamp_End)

                with socket.socket(socket.AF_INET, socket.SOCK_STREAM, 0) as sock:
                    sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)

                    sock.bind((self.HOST, self.PORT))

                    try:
                        sock.connect((self.SERVER_HOST, self.SERVER_PORT))

                        with self.context.wrap_socket(sock, server_hostname=self.HOST) as client:
                            if len(self.buffer_waiting) > 0:
                                pd_characts = pd.concat([self.buffer_waiting, pd_characts], ignore_index=True)
                                self.buffer_waiting = pd.DataFrame()

                            serialized_data = pickle.dumps(pd_characts, -1)
                            client.sendall(struct.pack(">I", len(serialized_data)))
                            client.sendall(serialized_data)
                            print("Данные успешно отправлены!")

                    except ConnectionRefusedError:
                        if len(self.buffer_waiting) == 0:
                            self.buffer_waiting = pd_characts
                        else:
                            self.buffer_waiting = pd.concat([self.buffer_waiting, pd_characts], ignore_index=True)
                            if len(self.buffer_waiting) > self.max_len_buffer:
                                self.buffer_waiting = self.buffer_waiting.iloc[-self.max_len_buffer:]
                        print("Не удалось установить соединение с сервером, данные сохранены в буфер")

            except Exception as err:
                logging.exception(f"Ошибка!\n{err}")

            return array_characts

    def run(self):
        if Path(self.path_name).exists():
            self.run_analyz = True

            print("Поток предварительного анализа событий процессов запущен")

            while self.run_analyz:
                count_file_events = self.GetFilesEvents()
                if count_file_events == 0:
                    time.sleep(10)
                    continue
                else:
                    try:
                        self.ProcessingEvents(self.files_events_arr)
                        self.files_events_arr.clear()
                    except IndexError:
                        continue

        else:
            print("Директория с файлами событий для анализа не существует")

    def stop(self):
        print("Поток предварительного анализа событий процессов завершён")
        self.run_analyz = False


if __name__ == '__main__':
    path_name                   = "D:\\train_dataset_Nout"
    thread_time_limit           = 1 * 50 * HUNDREDS_OF_NANOSECONDS
    charact_file_name           = "train_dataset_"
    user_dir                    = "Admin"

    analizator = AnalyzerEvents(thread_time_limit, charact_file_name, path_name, user_dir)
    analizator.start()
    analizator.join()