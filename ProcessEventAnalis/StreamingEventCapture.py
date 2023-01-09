from threading import Thread
from pathlib import Path

import subprocess as sp
import re


class SnifferEventProc(Thread):
    def __init__(self, size_pml_time, event_file_mask, path_procmon, path_procmon_config, path_name):
        super().__init__()

        self.size_pml_time          = size_pml_time
        self.event_file_mask        = event_file_mask
        self.path_name              = path_name
        self.path_procmon           = path_procmon
        self.path_procmon_config    = path_procmon_config

        self.tepm_path = f"{self.path_name}\\temp"
        if not Path(self.tepm_path).exists():
            Path(self.tepm_path).mkdir()

        self.last_file_id   = None
        self.th_main_sniff  = None
        self.run_sniff      = False

    def __StartProcmonSniff__(self, event_file):
        file_temp_name = f"{self.tepm_path}\\{event_file}"

        procmon = [self.path_procmon, "/BackingFile", file_temp_name,
                   "/Runtime", str(self.size_pml_time), "/Minimized",
                   "/LoadConfig", self.path_procmon_config]

        startupinfo = sp.STARTUPINFO()
        startupinfo.dwFlags |= sp.STARTF_USESHOWWINDOW

        prog = sp.Popen(procmon, startupinfo=startupinfo)
        prog.communicate()

        Path(file_temp_name).rename(f"{self.path_name}\\{event_file}")

    def GetLastFileId(self):
        path_sniffer_home = Path(self.path_name)
        file_arr = []

        for file in path_sniffer_home.iterdir():
            file = str(file)
            if not (self.event_file_mask in file):
                continue
            else:
                file_arr.append(file)

        if len(file_arr) > 0:
            indexs_files_pml = []

            # Получение индексов файлов с событиями процессов
            for file_name in file_arr:
                index = file_name.find(self.event_file_mask)
                index_file = [int(s) for s in re.split('_|.p', file_name[index:]) if s.isdigit()][0]
                indexs_files_pml.append(index_file)

            indexs_files_pml.sort()
            self.last_file_id = indexs_files_pml[-1]
        else:
            self.last_file_id = -1

    def SniffLoop(self):
        self.last_file_id += 1

        while self.run_sniff:
            event_file = f"{self.event_file_mask}{self.last_file_id}.pml"

            try:
                self.__StartProcmonSniff__(event_file)
                self.last_file_id += 1
            except Exception as err:
                print("Ошибка во время снифинга событий процессов! %s" % str(err))
                continue

    def run(self):
        if Path(self.path_name).exists():
            if not Path(self.path_name).exists():
                Path(self.path_name).mkdir()

            self.GetLastFileId()
            print(f"Индекс последнего файла с событиями процессов: {self.last_file_id}")

            print("Поток сбора событий процессов запущен")
            self.run_sniff = True
            self.SniffLoop()

        else:
            print("Директория для сбора событий процессов не существует, создайте её и перезапустите процесс!")

    def stop(self):
        print("Поток сбора трафика завершён")
        self.run_sniff = False


if __name__ == '__main__':
    # Параметры сборщика событий
    size_pml_time       = 10
    event_file_mask     = "event_log_"
    path_name           = "F:\\EVENT"
    path_procmon        = "Procmon64.exe"
    path_procmon_config = "ProcmonConfiguration.pmc"

    sniffer = SnifferEventProc(size_pml_time, event_file_mask, path_procmon, path_procmon_config, path_name)
    sniffer.run()