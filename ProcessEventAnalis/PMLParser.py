from ProcmonParser.consts import ColumnToOriginalName, Column, EventClass
from ProcmonParser import ProcmonLogsReader, Event

import matplotlib.pyplot as plt


class ParserEvents:
    """
        Класс генерации или получения информации обо всех событиях из файла PML,
        являющегося лог файлом программы Procmon.
        Принимает на вход строковое имя файла pml_file_name.
    """
    def __init__(self, pml_file_name:str):
        self.pml_file_name      = pml_file_name

        self.events             = []
        self.Anomaly_intensity  = []

    def GetEvents(self):
        """
            Функция получения массива со всеми событиями в файле pml.
            Возвращает массив со словарями описывающими каждое событие
        """
        with open(self.pml_file_name, "rb") as pml_file:
            pml_readers = ProcmonLogsReader(pml_file)
            self.events.clear()
            for pml_record in pml_readers:
                try:
                    event = self.GetEventInformation(pml_record)
                    self.events.append(event)
                except UnicodeEncodeError:
                    continue
        return self.events

    def GenEventIter(self):
        """
            Генератор событий. Возвращает на каждой итерации очередное событие из файла pml в виде словаря.
        """
        with open(self.pml_file_name, "rb") as pml_file:
            pml_readers = ProcmonLogsReader(pml_file)

            for pml_record in pml_readers:
                try:
                    event = self.GetEventInformation(pml_record)
                    yield event
                except UnicodeEncodeError:
                    continue

    def GetEventInformation(self, event: Event):
        """
            Возвращает информацию о событии в исходном виде, без преобразования в строковые интерпретации
        """

        record = {
            Column.DATE_AND_TIME: event.date_filetime,
            Column.PROCESS_NAME: event.process.process_name,
            Column.PID: int(event.process.pid),
            Column.EVENT_CLASS: event.event_class.value,
            Column.OPERATION: event.operation,
            Column.CATEGORY: event.category,
            Column.RESULT: event.result,
            Column.PATH: event.path,
            Column.DETAIL: self.GetDetailsEvent(event),
        }

        compatible_record = {ColumnToOriginalName[k]: v for k, v in record.items()}
        return compatible_record

    def GetDetailsEvent(self, event):
        """
            Возвращает столбец дополнительных сведений о событии в виде словаря
        """
        if not event.details:
            return dict()
        details = event.details.copy()
        necessary_details = {}
        # if EventClass.Registry == event.event_class:
        #     commas_formatted_keys = ["Length", "SubKeys", "Values"]
        #     for key in commas_formatted_keys:
        #         if key in details:
        #             necessary_details[key] = '{:,}'.format(details[key])

        if EventClass.File_System == event.event_class:
            commas_formatted_keys = ["AllocationSize", "Offset", "Length"]
            for key in commas_formatted_keys:
                if key in details and int == type(details[key]):
                    necessary_details[key] = '{:,}'.format(details[key])

        return dict(necessary_details)

    def GetAnomalyIntensity(self, AnomalyProcess, window_size):
        print("Запускаем анализ тестового набора аномалии")
        events_in_window = []
        self.Anomaly_intensity.clear()
        for i in range(len(self.events)):
            events_in_window.append(self.events[i])
            if len(events_in_window) < window_size:
                continue
            else:
                inten = 0
                for event in events_in_window:
                    if event["Process Name"] == AnomalyProcess:
                        inten += 1

                self.Anomaly_intensity.append(inten)
                events_in_window.pop(0)
        print("Анализ завершён")

        return self.Anomaly_intensity

    def PrintGrafAnomaly(self, scale_y):
        inten_max = max(self.Anomaly_intensity)

        print("Выполняем нормализацию от 0 до 100")
        for i in range(len(self.Anomaly_intensity)):
            self.Anomaly_intensity[i] = self.Anomaly_intensity[i] / inten_max * scale_y
        print("Нормализация Выполнена")

        plt.xlim([-10.0, len(self.Anomaly_intensity) + 10.0])
        plt.ylim([-5.0, 105.0])
        plt.title(f"График интенсивности аномальных событий в тестовом наборе событий процессов")
        plt.grid(which='major')
        plt.grid(which='minor', linestyle=':')

        plt.plot(self.Anomaly_intensity, label="Интенсивность аномалий", color="tab:red")

        plt.legend(fontsize=10)
        plt.tight_layout()
        plt.show()


def main():
    pml_file_name = "F:\\EVENT\\EventTest\\test_event.PML"
    anomaly_proc  = "RAT_client.exe"
    scale_y       = 100
    window_size   = 500

    print(f"Считываем события из тестового набора данных: {pml_file_name}")
    parser_pml = ParserEvents(pml_file_name)
    parser_pml.GetEvents()
    parser_pml.GetAnomalyIntensity(anomaly_proc, window_size)
    parser_pml.PrintGrafAnomaly(scale_y)


if __name__ == '__main__':
    main()