from threading import Thread
from pathlib import Path

import subprocess as sp
import re


class SnifferEventProc:
    def __init__(self, size_pml_time, event_name, event_file_mask, path_name):
        self.size_pml_time      = size_pml_time
        self.event_name         = event_name
        self.event_file_mask    = event_file_mask
        self.path_name          = path_name

        self.last_file_id   = None
        self.th_main_sniff  = None
        self.run_sniff      = False

    def __StartProcmonSniff(self, event_file):
        procmon = ['Procmon64.exe', '/BackingFile', event_file,
                   '/Runtime', str(self.size_pml_time), '/Minimized',
                   '/LoadConfig', 'ProcmonConfiguration.pmc']

        startupinfo = sp.STARTUPINFO()
        startupinfo.dwFlags |= sp.STARTF_USESHOWWINDOW

        prog = sp.Popen(procmon, startupinfo=startupinfo)
        prog.communicate()

    def GetLastFileId(self):
        path_sniffer_home = Path(self.path_name + "\\" + self.event_name)
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
                index = file_name.find(self.event_name)
                index_file = [int(s) for s in re.split('_|.p', file_name[index + 5:]) if s.isdigit()][0]
                indexs_files_pml.append(index_file)

            indexs_files_pml.sort()
            self.last_file_id = indexs_files_pml[-1]
        else:
            self.last_file_id = -1

    def SniffLoop(self):
        self.last_file_id += 1

        while self.run_sniff:
            event_file = f"{self.path_name}\\{self.event_name}\\{self.event_file_mask}{self.last_file_id}.pml"

            try:
                th_sniff = Thread(target=self.__StartProcmonSniff, args=(event_file,))
                th_sniff.start()
                th_sniff.join()

                self.last_file_id += 1
            except Exception as err:
                print("Ошибка во время снифинга событий процессов! %s" % str(err))
                continue

    def run(self):
        if Path(self.path_name).exists():
            if not Path(self.path_name + "\\" + self.event_name).exists():
                Path(self.path_name + "\\" + self.event_name).mkdir()

            self.GetLastFileId()
            print(f"Индекс последнего файла с событиями процессов: {self.last_file_id}")

            self.run_sniff = True
            self.th_main_sniff = Thread(target=self.SniffLoop, args=())
            self.th_main_sniff.start()
            print("Поток сбора событий процессов запущен")
        else:
            print("Директория для сбора событий процессов не существует, создайте её и перезапустите процесс!")

    def stop(self):
        print("Поток сбора трафика завершён")
        self.run_sniff = False


if __name__ == '__main__':
    # Параметры сборщика трафика
    size_pml_time   = 10
    event_name      = "EventTest"
    event_file_mask = "event_log_"
    path_name       = "F:\\EVENT"

    sniffer = SnifferEventProc(size_pml_time, event_name, event_file_mask, path_name)
    sniffer.run()