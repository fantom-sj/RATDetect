from ProcessEventAnalis.StreamingEventCapture import SnifferEventProc
from ProcessEventAnalis.AnalysisProcessEvents import AnalyzerEvents
from ProcessEventAnalis.EventsСharacts import HUNDREDS_OF_NANOSECONDS

from elevate import elevate


def main():
    path_event_analysis     = "WorkDirectory"
    size_pml_time           = 30
    event_file_mask         = "event_log_"
    path_procmon            = "ProcessEventAnalis\\Procmon64.exe"
    path_procmon_config     = "ProcessEventAnalis\\ProcmonConfiguration.pmc"

    thread_time_limit       = 1 * 50 * HUNDREDS_OF_NANOSECONDS
    user_dir                = "Жертва"
    max_len_buffer          = 1000

    HOST        = "192.168.10.128"
    PORT        = 62301
    SERVER_HOST = "192.168.137.1"
    SERVER_PORT = 62302
    cert        = "ca.crt"

    # sniffer = SnifferEventProc(size_pml_time, event_file_mask, path_procmon, path_procmon_config, path_event_analysis)
    # sniffer.start()

    analizator = AnalyzerEvents(thread_time_limit, path_event_analysis, user_dir, max_len_buffer,
                                HOST, PORT, SERVER_HOST, SERVER_PORT, cert)
    analizator.start()

    # analizator.DirectProcessingEvents("train_characts_events")

if __name__ == '__main__':
    # elevate()
    main()