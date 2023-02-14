eel.expose(receiverMonitStat);
function receiverMonitStat(monitor_proc_stat) {
    if (monitor_proc_stat == true) {
        $('.switch-btn').toggleClass('switch-on');
        console.log("Процесс мниторинга снова включен!");
    }
}

function DataAnalysisPrintRes(buffer_res) {
    var netflows    = buffer_res.NetFlows;
    var process     = buffer_res.Process;
    var rat_trojans = buffer_res.RATtrojans;

    $.each(netflows, function(flow, flow_active) {
        Append_NetFlowAnalyseRes(flow, flow_active);
    });
    $.each(process, function(proc, proc_active) {
        Append_ProcessAnalyseRes(proc, proc_active);
    });
    $.each(rat_trojans, function(rat, rat_active) {
        Append_RATAnalyseRes(rat, rat_active);
    });
}

function Append_NetFlowAnalyseRes(flow, flow_active) {
    if (flow_active.anomal_active > 0) {
        var ips_port = flow.substring(11, flow.length-2).split(", ");

        var ip1      = undefined;
        var ip2      = undefined;
        var port     = undefined;

        for (let id in ips_port) {
            var flow_part = ips_port[id]
            if (flow_part.length > 5) {
                if (ip1 == undefined) {
                    ip1 = Number(flow_part);
                }
                else {
                    ip2 = Number(flow_part);
                }
            }
            else {
                port = Number(flow_part);
            }
        }

        UpdateTblNetFlowRes(ip1, ip2, port, flow_active.anomal_active, flow_active.normal_active)
    }
}

function Append_ProcessAnalyseRes(proc, proc_active) {
    var proc_name     = proc;
    var anomal_active = proc_active.anomal_active;
    var normal_active = proc_active.normal_active;
    if (anomal_active > 0) {
        var proc_connects = [];
        if (proc_active.netflows.length > 0) {
            proc_connects = proc_active.netflows;
        }
        UpdateTblProcessRes(proc_name, proc_connects, anomal_active, normal_active);
    }
}

function Append_RATAnalyseRes(rat, rat_active) {
    var rat_name   = undefined;
    var rat_device = undefined;

    if ("PossibilityRAT" in rat_active) {
        if (rat_active.PossibilityRAT > 0) {
            var rat_info = rat.substring(11, rat.length-2).split(", ");
            if (rat_info[0].includes(".exe")) {
                rat_name   = rat_info[0].substring(1, rat_info[0].length-1)
                rat_device = rat_info[1].substring(1, rat_info[1].length-1)
            }
            else {
                rat_name   = rat_info[1].substring(1, rat_info[1].length-1)
                rat_device = rat_info[0].substring(1, rat_info[0].length-1)
            }

            /*console.log(rat_info[1] + " " + rat_info[0] + " " + rat_active["PercentAnomalyEvents"] + " " +
                        rat_active["PercentAnomalyNetFlow"] + " " + rat_active["PossibilityRAT"])*/
            UpdateTblRATRes(rat_device, rat_name, rat_active["PercentAnomalyEvents"],
                            rat_active["PercentAnomalyNetFlow"], rat_active["PossibilityRAT"]);
        }
    }
}

function UpdateTblNetFlowRes(ip1, ip2, port, anomal_active, normal_active) {
    var id_flow = ip1 + "_" + ip2 + "_" + port;

    if ($(`#${id_flow}_netflow`).length > 0) {
        var old_anomal_active = $(`#${id_flow}_anomal_active`).text()
        var old_normal_active = $(`#${id_flow}_normal_active`).text()

        if (old_anomal_active != anomal_active) {
            $(`#${id_flow}_anomal_active`).animate({
                backgroundColor: "#FD3050"
            }, 300 );
            $(`#${id_flow}_anomal_active`).text(anomal_active);
            $(`#${id_flow}_anomal_active`).animate({
                backgroundColor: "#f4f6f9"
            }, 300 );
        }
        if (old_normal_active != normal_active) {
            $(`#${id_flow}_normal_active`).animate({
                backgroundColor: "#6cf369"
            }, 300 );
            $(`#${id_flow}_normal_active`).text(normal_active);
            $(`#${id_flow}_normal_active`).animate({
                backgroundColor: "#f4f6f9"
            }, 300 );
        }
    }
    else {
        var row = `<td id="${id_flow}_netflow" width="40%" style="text-align: left; vertical-align: middle;">
                        IP1: ${IPnum2string(ip1)}, IP2: ${IPnum2string(ip2)}, Порт: ${port}
                   </td>`;
            row += `<td id="${id_flow}_anomal_active" width="30%" style="text-align: center; vertical-align: middle;">
                        ${anomal_active}
                    </td>`;
            row += `<td id="${id_flow}_normal_active" width="30%" style="text-align: center; vertical-align: middle;">
                        ${normal_active}
                    </td>`;

        var new_row = `<tr id="${id_flow}">${row}</tr>`
        $('#AnomalyNetFlow').append(new_row);
    }
}

function UpdateTblProcessRes(proc_name, proc_connects, anomal_active, normal_active) {
    var id_proc = proc_name.split(".")[0];

//    console.log(proc_name + " " + anomal_active + " " + normal_active);
    if ($(`#${id_proc}_process`).length > 0) {
        var old_anomal_active = Number($(`#${id_proc}_anomal_active`).text())
        var old_normal_active = Number($(`#${id_proc}_normal_active`).text())

        if (old_anomal_active != anomal_active) {
            $(`#${id_proc}_anomal_active`).animate({
                backgroundColor: "#FD3050"
            }, 300 );
            $(`#${id_proc}_anomal_active`).text(anomal_active);
            $(`#${id_proc}_anomal_active`).animate({
                backgroundColor: "#f4f6f9"
            }, 300 );
        }
        if (old_normal_active != normal_active) {
            $(`#${id_proc}_normal_active`).animate({
                backgroundColor: "#6cf369"
            }, 300 );
            $(`#${id_proc}_normal_active`).text(normal_active);
            $(`#${id_proc}_normal_active`).animate({
                backgroundColor: "#f4f6f9"
            }, 300 );
        }
    }
    else {
        var row = `<td id="${id_proc}_process" width="40%" style="text-align: left; vertical-align: middle;">
                        ${proc_name}
                    </td>`;
        row += `<td id="${id_proc}_anomal_active" width="30%" style="text-align: center; vertical-align: middle;">
                    ${anomal_active}
                </td>`;
        row += `<td id="${id_proc}_normal_active" width="30%" style="text-align: center; vertical-align: middle;">
                    ${normal_active}
                </td>`;

        var new_row = `<tr id="${id_proc}">${row}</tr>`;
        $('#AnomalyProcess').append(new_row);
    }
}

function UpdateTblRATRes(rat_device, rat_name, PercentAnomalyEvents, PercentAnomalyNetFlow, PossibilityRAT) {
    var id_rat = rat_device.split(".").join("_") + "_" + rat_name.substring(0, rat_name.length-4)
    var anomaly_events  = Math.round(PercentAnomalyEvents * 100) / 100
    var anomaly_netflow = Math.round(PercentAnomalyNetFlow * 100) / 100
    var possibility_rat = Math.round(PossibilityRAT * 1000) / 1000

    if ($(`#${id_rat}`).length > 0) {
        console.log("Заменили");
        var old_anomaly_events  = Math.floor($(`#${id_rat}_anomaly_events`).text())
        var old_anomaly_netflow = Math.floor($(`#${id_rat}_anomaly_netflow`).text())
        var old_possibility_rat = Math.floor($(`#${id_rat}_possibility_rat`).text())

        if (old_anomaly_events != anomaly_events) {
            if (anomal_active > old_anomaly_events) {
                $(`#${id_rat}_anomaly_events`).animate({
                    backgroundColor: "#FD3050"
                }, 300 );
                $(`#${id_rat}_anomaly_events`).text(anomal_active);
                $(`#${id_rat}_anomaly_events`).animate({
                    backgroundColor: "#f4f6f9"
                }, 300 );
            }
            else {
                $(`#${id_rat}_anomaly_events`).animate({
                    backgroundColor: "#6cf369"
                }, 300 );
                $(`#${id_rat}_anomaly_events`).text(anomal_active);
                $(`#${id_rat}_anomaly_events`).animate({
                    backgroundColor: "#f4f6f9"
                }, 300 );
            }
        }

        if (old_anomaly_netflow != anomaly_netflow) {
            if (anomaly_netflow > old_anomaly_netflow) {
                $(`#${id_rat}_anomaly_netflow`).animate({
                    backgroundColor: "#FD3050"
                }, 300 );
                $(`#${id_rat}_anomaly_netflow`).text(anomaly_netflow);
                $(`#${id_rat}_anomaly_netflow`).animate({
                    backgroundColor: "#f4f6f9"
                }, 300 );
            }
            else {
                $(`#${id_rat}_anomaly_netflow`).animate({
                    backgroundColor: "#6cf369"
                }, 300 );
                $(`#${id_rat}_anomaly_netflow`).text(anomaly_netflow);
                $(`#${id_rat}_anomaly_netflow`).animate({
                    backgroundColor: "#f4f6f9"
                }, 300 );
            }
        }

        if (old_possibility_rat != possibility_rat) {
            if (possibility_rat > old_possibility_rat) {
                $(`#${id_rat}_possibility_rat`).animate({
                    backgroundColor: "#FD3050"
                }, 300 );
                $(`#${id_rat}_possibility_rat`).text(possibility_rat);
                $(`#${id_rat}_possibility_rat`).animate({
                    backgroundColor: "#f4f6f9"
                }, 300 );
            }
            else {
                $(`#${id_rat}_possibility_rat`).animate({
                    backgroundColor: "#6cf369"
                }, 300 );
                $(`#${id_rat}_possibility_rat`).text(possibility_rat);
                $(`#${id_rat}_possibility_rat`).animate({
                    backgroundColor: "#f4f6f9"
                }, 300 );
            }
        }
    }
    else {

        console.log("Добавили")
        var row = `<td id="${id_rat}_device" width="18%" style="text-align: left; vertical-align: middle;">
                       ${rat_device}
                   </td>`;
        row += `<td id="${id_rat}_process" width="18%" style="text-align: center; vertical-align: middle;">
                    ${rat_name}
                </td>`;
        row += `<td id="${id_rat}_anomaly_events" width="21%" style="text-align: center; vertical-align: middle;">
                    ${anomaly_events}
                </td>`;
        row += `<td id="${id_rat}_anomaly_netflow" width="21%" style="text-align: center; vertical-align: middle;">
                    ${anomaly_netflow}
                </td>`;
        row += `<td id="${id_rat}_possibility_rat" width="26%" style="text-align: center; vertical-align: middle;">
                    ${possibility_rat}
                </td>`;
        var new_row = `<tr id="${id_rat}">${row}</tr>`;
        $('#RAT_records').append(new_row);
    }
}