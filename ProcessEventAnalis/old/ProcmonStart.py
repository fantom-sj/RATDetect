from ProcmonParser import load_configuration, dump_configuration, Rule
from threading import Thread
import subprocess as sp


def __StartProcmon(index_log, time):
    prog = sp.Popen(['Procmon64.exe', '/BackingFile', 'log_pml/Log' +
                     str(index_log) + '.pml', '/Runtime', str(time), '/Minimized',
                     '/LoadConfig', 'ProcmonConfiguration.pmc'], stdin=sp.PIPE)
    prog.communicate()


def __StartConvertInCsv(index_csv):
    prog = sp.Popen(['Procmon64.exe', '/OpenLog', 'log_pml/Log' +
                     str(index_csv) + '.pml', '/SaveAs', 'log_csv/Log' +
                     str(index_csv) + '.csv'], stdin=sp.PIPE)
    prog.communicate()


def StartParse(proc_pid=-1, index_parce=0, time=-1):
    with open('../ProcmonConfiguration.pmc', 'rb') as f:
        config = load_configuration(f)

    rule = [Rule('PID', 'is', str(proc_pid), 'include')]
    config["FilterRules"] = rule
    config["DestructiveFilter"] = 1

    with open('../ProcmonConfiguration.pmc', 'wb') as file:
        dump_configuration(config, file)

    th_log = Thread(target=__StartProcmon, args=(index_parce, time,))
    th_convert = Thread(target=__StartConvertInCsv, args=(index_parce,))

    th_log.start()
    while th_log.is_alive():
        continue

    th_convert.start()





if __name__ == '__main__':
    main()