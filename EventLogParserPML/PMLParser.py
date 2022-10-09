from procmon_parser import Event
from procmon_parser import ProcmonLogsReader
from procmon_parser.consts import ColumnToOriginalName, Column, EventClass


class ParserEventInPML:
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
        pml_readers = ProcmonLogsReader(open(self.pml_file_name, "rb"))
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
        pml_readers = ProcmonLogsReader(open(self.pml_file_name, "rb"))

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
        if EventClass.Registry == event.event_class:
            commas_formatted_keys = ["Length", "SubKeys", "Values"]
            for key in commas_formatted_keys:
                if key in details:
                    necessary_details[key] = '{:,}'.format(details[key])

        elif EventClass.File_System == event.event_class:
            commas_formatted_keys = ["AllocationSize", "Offset", "Length"]
            for key in commas_formatted_keys:
                if key in details and int == type(details[key]):
                    necessary_details[key] = '{:,}'.format(details[key])

        return dict(necessary_details)


def main():
    pml_file_name = "log_pml\\Log2.pml"
    parser_pml = ParserEventInPML(pml_file_name)
    events = parser_pml.GetEvents()

    for event in parser_pml.GenEventIter():
        print(event)


if __name__ == '__main__':
    main()