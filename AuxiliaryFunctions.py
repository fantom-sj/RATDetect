from pathlib import Path


def GetFilesCSV(path):
    path = Path(path)
    files_local = {}
    files_timecreate = []

    for file in path.iterdir():
        file_split = str(file).split(".")

        if file.is_file():
            if file_split[1] == "csv" or file_split[1] == "CSV":
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

    files_characts_arr = []
    if len(files_timecreate) > 0:
        files_timecreate.sort()

        for tc in files_timecreate:
            files_characts_arr.append(files_local[tc])
        return files_characts_arr
    else:
        return []