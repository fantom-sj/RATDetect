from ProcmonParser import Event
from ProcmonParser import ProcmonLogsReader
from ProcmonParser.consts import ColumnToOriginalName, Column, EventClass
from EventsСharacts import CulcCharactsEventsOnWindow

class ParserEvents:
    """
        Класс генерации или получения информации обо всех событиях из файла PML,
        являющегося лог файлом программы Procmon.
        Принимает на вход строковое имя файла pml_file_name.
    """
    def __init__(self, pml_file_name:str):
        self.pml_file_name = pml_file_name

    def GetEvents(self):
        """
            Функция получения массива со всеми событиями в файле pml.
            Возвращает массив со словарями описывающими каждое событие
        """
        with open(self.pml_file_name, "rb") as pml_file:
            pml_readers = ProcmonLogsReader(pml_file)
            events = []
            for pml_record in pml_readers:
                try:
                    event = self.GetEventInformation(pml_record)
                    events.append(event)
                except UnicodeEncodeError:
                    continue
        return events

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


def main():
    pml_file_name = "F:\\EVENT\\EventTest\\event_log_8.pml"
    parser_pml = ParserEvents(pml_file_name)
    events = parser_pml.GetEvents()

    window_size = 1000

    for i in range(0, len(events)-window_size, window_size):
        print(CulcCharactsEventsOnWindow(events[i:i+window_size], window_size))


if __name__ == '__main__':
    main()