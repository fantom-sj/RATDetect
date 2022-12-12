from ProcessEventAnalis.StreamingEventCapture import SnifferEventProc
from ProcessEventAnalis.AnalysisProcessEvents import AnalyzerEvents

from elevate import elevate


def main():
    path_event_analysis     = "C:\\Users\\Жертва\\Desktop\\RATDetect"
    size_pml_time           = 20
    event_file_mask         = "event_log_"
    path_procmon            = "ProcessEventAnalis\\Procmon64.exe"
    path_procmon_config     = "ProcessEventAnalis\\ProcmonConfiguration.pmc"

    window_size             = 50
    charact_file_name       = "events_characters_"
    user_dir                = "Жертва"

    sniffer = SnifferEventProc(size_pml_time, event_file_mask,
                               path_procmon, path_procmon_config, path_event_analysis)
    sniffer.run()

    analizator = AnalyzerEvents(window_size, charact_file_name,
                                path_event_analysis, user_dir)

    analizator.run()


if __name__ == '__main__':
    elevate()
    main()