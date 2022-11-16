from ProcessEventAnalis.EventsСharacts import CulcCharactsEventsOnWindow, CHARACTERISTIC_EVENTS
from PMLParser import ParserEvents
from threading import Thread
from pathlib import Path


import subprocess as sp
import pandas as pd

import logging
import time
import dpkt
import math
import re

class AnalyzerEvents:
    def __init__(self, window_size, charact_file_length, charact_file_mask, event_name, path_name):
        self.window_size            = window_size
        self.charact_file_length    = charact_file_length
        self.charact_file_mask      = charact_file_mask
        self.event_name             = event_name
        self.path_name              = path_name

        self.files_events_arr       = []
        self.array_events_global     = []

        self.run_analyz             = True
        self.th_main_analyz         = None
        self.index_charact_file     = 0

        self.GetLastFileId()
        self.GetFilesEvents()

    def GetFilesEvents(self):
        path_home = Path(self.path_name + "\\" + self.event_name)
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
        path_home = Path(self.path_name + "\\" + self.event_name)
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
            self.index_charact_file = 0

    def ProcessingEvents(self, pml_file_name=None):
        if not (pml_file_name is None):
            print(f"Запускаем обработку файла: {pml_file_name}")
            parser = ParserEvents(pml_file_name)
            for event in parser.GenEventIter():
                self.array_events_global.append(event)

            path_new = self.path_name + "\\" + self.event_name + "\\" + "Обработанные файлы"
            if not Path(path_new).exists():
                Path(path_new).mkdir()
            file_only_name = pml_file_name.split("\\")[-1]
            Path(pml_file_name).rename(path_new + "\\" + file_only_name)

        array_characts = []
        try:
            while len(self.array_events_global) >= self.window_size:
                array_pkt = self.array_events_global[:self.window_size]
                ch = CulcCharactsEventsOnWindow(array_pkt, self.window_size)
                if ch is not None:
                    array_characts.append(ch)
                else:
                    continue
                self.array_events_global.pop(0)
                # for i in range(self.window_size):
                #     self.array_events_global.pop(0)

        except Exception as err:
            logging.exception(f"Ошибка!\n{err}")
            return False

        print("Выявлено характеристик: %d" % len(array_characts))

        if len(array_characts) == 0:
            print("Не выделено ни одного набора характеристик!")
            return False
        else:
            try:
                characts_file_name = self.path_name + "\\" + self.event_name + "\\" + \
                                     self.charact_file_mask + str(self.index_charact_file) + ".csv"

                pd_characts_old = pd.read_csv(characts_file_name)
                pd_characts = pd.DataFrame(array_characts)

                pd_characts_new = pd.concat([pd_characts_old, pd_characts], ignore_index=False)
                pd_characts_arr = []
                num_chunks = math.ceil(len(pd_characts_new) / self.charact_file_length)
                for i in range(num_chunks):
                    pd_characts_arr.append(pd_characts_new[i * self.charact_file_length:(i + 1) * self.charact_file_length])

                pd_characts_arr[0].to_csv(characts_file_name, index=False)
                if len(pd_characts_arr) == 2:
                    self.index_charact_file += 1
                    characts_file_name = self.path_name + "\\" + self.event_name + "\\" + \
                                         self.charact_file_mask + str(self.index_charact_file) + ".csv"
                    pd_characts_arr[1].to_csv(characts_file_name, index=False)

                print("Парсинг завершился!")
                return True
            except Exception as err:
                logging.exception(f"Ошибка!\n{err}")
                return False

    def AnalyzLoop(self):

        while self.run_analyz:
            count_file_events = self.GetFilesEvents()
            if count_file_events == 0:
                time.sleep(30)
                continue
            else:
                try:
                    self.ProcessingEvents(self.files_events_arr[0])
                    self.files_events_arr.pop(0)
                except IndexError:
                    continue

    def run(self):
        if Path(self.path_name + "\\" + self.event_name).exists():
            characts_file_name = self.path_name + "\\" + self.event_name + "\\" + \
                                 self.charact_file_mask + str(self.index_charact_file) + ".csv"
            if not Path(characts_file_name).exists():
                pd_ch_name = pd.DataFrame()
                for ch in CHARACTERISTIC_EVENTS:
                    pd_ch_name[ch] = []
                pd_ch_name.to_csv(str(characts_file_name), index=False)

            self.run_analyz = True
            self.th_main_analyz = Thread(target=self.AnalyzLoop, args=())
            self.th_main_analyz.start()
            print("Поток анализа событий процессов запущен")

        else:
            print("Директория с файлами событий для анализа не существует")


if __name__ == '__main__':
    events_name                 = "EventTest"
    path_name                   = "F:\\EVENT"
    window_size                 = 500
    charact_file_length         = 1000000
    charact_file_name           = "dataset_"

    analizator = AnalyzerEvents(window_size, charact_file_length,
                                charact_file_name, events_name, path_name)
    analizator.run()