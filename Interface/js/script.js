const EPOCH_AS_FILETIME       = 116444736000000000;
const HUNDREDS_OF_NANOSECONDS = 10000000;

var type_data             = undefined;
var block                 = false;
var LineChartEvents       = undefined;
var LineChartNetflow      = undefined;
var max_records           = 0;

eel.expose(receiver);
function receiver(buffer_json, type) {
    //console.log("Вывод принятых данных");

    while (block) {
        setTimeout(function(){
            console.log("Ждём загрузки предыдущих данных");
        }, 100);
    }

    const data = JSON.parse(buffer_json);
    //console.log(buffer_output);

    if ($("#DataAnalysisRes").length > 0 && type == "data_analys_res"){
        DataAnalysisPrintRes(data);
    }
    else if ($("#AnalyseProcessEvents").length > 0 && type == "data_events"){
        AnalyseProcessEvents(data);
    }
    else if ($("#AnalyseProcessNetflow").length > 0 && type == "data_netflow"){
        AnalyseProcessNetflow(data)
    }
}

function AnalyseProcessEvents(buffer_events) {
	// Добавление в таблицу
    if (length in buffer_events) {
        if (buffer_events.length > 0) {
            data_in_graf = [];
            for (let record of buffer_events) {
                Append_tr_Events(record);
                data_in_graf.push([record.ID, record.loss]);
            }
            appendPointInGraphEvents(data_in_graf);
        }
    }
}

function AnalyseProcessNetflow(buffer_traffic) {
	// Добавление в таблицу
    if (length in buffer_traffic) {
        if (buffer_traffic.length > 0) {
            data_in_graf = [];
            for (let record of buffer_traffic) {
                Append_tr_NetFlow(record);
                data_in_graf.push([record.ID, record.loss]);
            }
            appendPointInGraphNetflow(data_in_graf);
        }
    }
}

function Append_tr_Events(record){
    var row = `<tr `;

    if ('anomaly' in record) {
        if (record.anomaly == 1){
            row += `style="background: #FD3050; color: #FFFFFF; font-weight: bold;"`;
        }

        row += `><td width="5%">${record.ID}</td>`;
        row += `<td width="20%">${timeConverter(record.Time_Stamp_Start)}</td>`;
        row += `<td width="25%">${timeConverter(record.Time_Stamp_End)}</td>`;
        row += `<td width="25%">${record.Process_Name}</td>`;

        var conns = ``;
        if ('connection' in record) {
            if (record.connection != 0) {
                for (let i = 0; i < record.connection.length; i++) {
                    var ip   = undefined;
                    var port = undefined
                    if (record.connection[i].length == 3) {
                        ip   = record.connection[i][1];
                        port = record.connection[i][2];
                    }
                    else {
                        ip   = record.connection[i][0];
                        port = record.connection[i][1];
                    }
                    console.log(record.connection[i]);
                    if (i == 0) {
                        conns += `IP: ${IPnum2string(ip)}`
                        conns += ` Порт: ${port}`;
                    }
                    else {
                        conns += `<br>IP: ${IPnum2string(ip)}`;
                        conns += ` Порт: ${port}`;
                    }
                }
            }
        }

        row += `<td width="25%">${conns}</td></tr>`;

        $('#results').append(row);
    }
    if($('#results tr').length >= 10) {
        $('#records').addClass('add-scroll');
    }
}

function Append_tr_NetFlow(record) {
    var row = `<tr `

    if ('anomaly' in record) {
        if (record.anomaly == 1) {
            row += `style="background: #FD3050; color: #FFFFFF; font-weight: bold;"`;
        }

        row += `><td width="5%">${record.ID}</td>`;
        row += `<td width="22.5%">${timeConverter(record.Time_Stamp_Start)}</td>`;
        row += `<td width="22.5%">${timeConverter(record.Time_Stamp_End)}</td>`;

        var conns_src = `IP: ${IPnum2string(record.Src_IP_Flow)}`;
        conns_src    += ` Порт: ${record.Src_Port_Flow}`;
        row += `<td width="25%">${conns_src}</td>`;

        var conns_dst = `IP: ${IPnum2string(record.Dst_IP_Flow)}`;
        conns_dst    += ` Порт: ${record.Dst_Port_Flow}`;
        row += `<td width="25%">${conns_dst}</td></tr>`;

        $('#results').append(row);
    }
    if($('#results tr').length >= 10) {
        $('#records').addClass('add-scroll');
    }
}

function timeConverter(timestamp){
  var a = new Date((timestamp - EPOCH_AS_FILETIME) / HUNDREDS_OF_NANOSECONDS * 1000);
  var months = ['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec'];
  var year = a.getFullYear();
  var month = months[a.getMonth()];
  var date = a.getDate();
  var hour = a.getHours();
  var min = a.getMinutes();
  var sec = a.getSeconds();
  var time = date + ' ' + month + ' ' + year + ' ' + hour + ':' + min + ':' + sec;
  return time;
}

eel.expose(receiverDataCash);
function receiverDataCash(type_data, data_cash) {
    block = true;
    if (type_data == "data_events") {
        AnalyseProcessEvents(data_cash);
        block = false;
    }
    else if (type_data == "data_netflow") {
        AnalyseProcessNetflow(data_cash);
        block = false;
    }
    else{
        block = false;
    }
}

eel.expose(SetMaxRecords);
function SetMaxRecords(max_r){
    max_records = max_r;
    if (max_r == 0)
        set_max = "all"
    else
        set_max = max_r
    $("#count_records").val(set_max).change();
}

function IPnum2string(ip) {
    return [24,16,8,0].map(n => (ip >> n) & 0xff).join(".")
}
