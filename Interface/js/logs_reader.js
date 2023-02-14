log_files_events  = []
log_files_netflow = []

eel.expose(set_files_log);
function set_files_log(files_log, type_log) {
    for (file in files_log) {
        if (type_log == "events") {
            $('#select_events').append(`<option value=${files_log[file]}>${files_log[file]}</option>`);
        }
        else {
            $('#select_netflows').append(`<option value=${files_log[file]}>${files_log[file]}</option>`);
        }
    }
}

eel.expose(set_log_data);
function set_log_data(log_data, type_log) {
    console.log(log_data);
    if ("events" == type_log) {
        for (idx in log_data){
            var row = `<td width="5%" style="text-align: left; vertical-align: middle;">
                           ${log_data[idx]["ID"]}
                       </td>`;
            row += `<td width="15%" style="text-align: center; vertical-align: middle;">
                        ${timeConverter(log_data[idx]["Time_Stamp_Start"])}
                    </td>`;
            row += `<td width="15%" style="text-align: center; vertical-align: middle;">
                        ${timeConverter(log_data[idx]["Time_Stamp_End"])}
                    </td>`;
            row += `<td width="15%" style="text-align: center; vertical-align: middle;">
                        ${log_data[idx]["Process_Name"]}
                    </td>`;
            row += `<td width="25%" style="text-align: center; vertical-align: middle;">
                        ${log_data[idx]["connection"]}
                    </td>`;
            row += `<td width="20%" style="text-align: center; vertical-align: middle;">
                        ${log_data[idx]["loss"]}
                    </td>`;
            var new_row = `<tr>${row}</tr>`;
            $('#tbl_events').append(new_row);
        }
    }
    else {
        for (idx in log_data){
            var row = `<td width="5%" style="text-align: left; vertical-align: middle;">
                           ${log_data[idx]["ID"]}
                       </td>`;
            row += `<td width="17.5%" style="text-align: center; vertical-align: middle;">
                        ${timeConverter(log_data[idx]["Time_Stamp_Start"])}
                    </td>`;
            row += `<td width="17.5%" style="text-align: center; vertical-align: middle;">
                        ${timeConverter(log_data[idx]["Time_Stamp_End"])}
                    </td>`;
            row += `<td width="20%" style="text-align: center; vertical-align: middle;">
                        IP: ${IPnum2string(log_data[idx]["Src_IP_Flow"])} Порт: ${log_data[idx]["Src_Port_Flow"]}
                    </td>`;
            row += `<td width="20%" style="text-align: center; vertical-align: middle;">
                        IP: ${IPnum2string(log_data[idx]["Dst_IP_Flow"])} Порт: ${log_data[idx]["Dst_Port_Flow"]}
                    </td>`;
            row += `<td width="20%" style="text-align: center; vertical-align: middle;">
                        ${log_data[idx]["loss"]}
                    </td>`;
            var new_row = `<tr>${row}</tr>`;
            $('#tbl_netflows').append(new_row);
        }
    }

}