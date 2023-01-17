import eel
from threading import Thread


class Interface(Thread):
    def __init__(self):
        super().__init__()

        self.data_cash = dict()
        self.data_cash["traffic"] = list()
        self.data_cash["events"] = list()

    def run(self):
        eel.init("Interface")
        eel.start("index.html", mode="opera")

    def SendinData(self, buffer_output_json):
        # print("Выводим данные в форму")
        eel.receiver(buffer_output_json)


interface = Interface()


@eel.expose
def SendDataCash(type_data):
    if type_data == "data_events" and len(interface.data_cash["events"]) > 0:
        eel.receiverDataCash(type_data, interface.data_cash["events"])
    elif type_data == "data_netflow" and len(interface.data_cash["traffic"]) > 0:
        eel.receiverDataCash(type_data, interface.data_cash["traffic"])