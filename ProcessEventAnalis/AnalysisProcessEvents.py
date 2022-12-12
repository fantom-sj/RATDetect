from ProcessEventAnalis.EventsСharacts import CulcCharactsEventsOnWindow, CHARACTERISTIC_EVENTS
from ProcessEventAnalis.PMLParser import ParserEvents
from threading import Thread
from pathlib import Path
from tqdm import tqdm

import pandas as pd
import logging
import time
import math


class AnalyzerEvents:
    def __init__(self, window_size, charact_file_mask, path_name, user_dir):
        self.window_size            = window_size
        self.charact_file_mask      = charact_file_mask
        self.path_name              = path_name
        self.user_dir               = user_dir

        self.files_events_arr       = []
        self.array_events_global    = []
        self.process_arr            = []

        self.run_analyz             = True
        self.th_main_analyz         = None
        self.index_charact_file     = -1

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

    def ProcessingEvents(self, pml_file_name=None):
        if not (pml_file_name is None):
            print(f"Запускаем обработку файла: {pml_file_name}")
            parser = ParserEvents(pml_file_name)

            pbar = tqdm(desc="Загрузка событий")
            for event in parser.GenEventIter():
                self.array_events_global.append(event)
                pbar.update(1)
            pbar.close()

            path_new = self.path_name + "\\" + "Обработанные файлы"
            if not Path(path_new).exists():
                Path(path_new).mkdir()
            file_only_name = pml_file_name.split("\\")[-1]
            Path(pml_file_name).rename(path_new + "\\" + file_only_name)

        array_characts  = []

        def analyz_thread_func(batch):
            ch = CulcCharactsEventsOnWindow(batch, self.user_dir)
            if ch is not None:
                array_characts.append(ch)

        thread_arr = []
        batch_events_process = {}
        try:
            pbar = tqdm(total=len(self.array_events_global), desc="Анализ событий")
            idx = 0
            while idx < len(self.array_events_global):
                event = self.array_events_global[idx]
                process_name = event["Process Name"]
                # if "1.exe" in process_name:
                #     idx += 1
                #     pbar.update(1)
                #     continue

                if not process_name in batch_events_process:
                    batch_events_process[process_name] = []
                batch_events_process[process_name].append(event)
                idx += 1
                pbar.update(1)
                if idx < len(self.array_events_global):
                    while self.array_events_global[idx]["Process Name"] == process_name:
                        # if "1.exe" in self.array_events_global[idx]["Process Name"]:
                        #     idx += 1
                        #     pbar.update(1)
                        #     continue
                        batch_events_process[process_name].append(self.array_events_global[idx])
                        idx += 1
                        if idx == len(self.array_events_global):
                            break
                        pbar.update(1)
                        if len(batch_events_process[process_name]) == self.window_size:
                            thread_arr.append(Thread(target=analyz_thread_func,
                                                     args=(batch_events_process[process_name],)))
                            thread_arr[-1].start()
                            batch_events_process[process_name].clear()

                if len(batch_events_process[process_name]) > 5:
                    thread_arr.append(Thread(target=analyz_thread_func, args=(batch_events_process[process_name],)))
                    thread_arr[-1].start()
                    batch_events_process[process_name].clear()

            for proc_events in batch_events_process:
                if len(batch_events_process[proc_events]) > 0:
                    thread_arr.append(Thread(target=analyz_thread_func, args=(batch_events_process[proc_events],)))
                    thread_arr[-1].start()
                    batch_events_process[proc_events].clear()

            control = True
            while control:
                control = False
                for thr in thread_arr:
                    control |= thr.is_alive()
                print("Ждем окончания анализа...")

            pbar.close()

        except Exception as err:
            logging.exception(f"Ошибка!\n{err}")
            return {}

        self.array_events_global.clear()
        print(f"Выявлены {len(array_characts)} характеристики событий процессов")

        if len(array_characts) == 0:
            print("Не выявлено ни одного набора характеристик!")
            return False
        else:
            try:
                # self.GetLastFileId()
                self.index_charact_file += 1
                characts_file_name = self.path_name + "\\" + \
                                     self.charact_file_mask + str(self.index_charact_file) + ".csv"

                pd_characts = pd.DataFrame(array_characts)
                pd_characts = pd_characts.sort_values(by="Time_Stamp_End")
                pd_characts.to_csv(characts_file_name, index=False)

                print("Парсинг завершился!")

            except Exception as err:
                logging.exception(f"Ошибка!\n{err}")

            return array_characts

    def AnalyzLoop(self):

        while self.run_analyz:
            count_file_events = self.GetFilesEvents()
            if count_file_events == 0:
                time.sleep(10)
                continue
            else:
                try:
                    self.ProcessingEvents(self.files_events_arr[0])
                    self.files_events_arr.pop(0)
                except IndexError:
                    continue

    def run(self):
        if Path(self.path_name).exists():
            self.run_analyz = True
            self.th_main_analyz = Thread(target=self.AnalyzLoop, args=())
            self.th_main_analyz.start()
            print("Поток предварительного анализа событий процессов запущен")

        else:
            print("Директория с файлами событий для анализа не существует")

    def stop(self):
        print("Поток предварительного анализа событий процессов завершён")
        self.run_analyz = False


if __name__ == '__main__':
    path_name                   = "F:\\EVENT"
    window_size                 = 50
    charact_file_name           = "test_dataset_1m_RAT_"
    user_dir                    = "Жертва"

    analizator = AnalyzerEvents(window_size, charact_file_name, path_name, user_dir)
    analizator.run()