const EPOCH_AS_FILETIME       = 116444736000000000
const HUNDREDS_OF_NANOSECONDS = 10000000

var buffer_output_events = []
var buffer_output_traffic = []
var block                = false
var LineChartEvents      = undefined
var LineChartNetflow     = undefined

eel.expose(receiver);
function receiver(buffer_output_json) {
    //console.log("Вывод принятых данных");

    while (block) {
        SetTimeout(function(){
            console.log("Ждём загрузки предыдущих данных");
        }, 100);
    }

    const buffer_output = JSON.parse(buffer_output_json);
    //console.log(buffer_output);

    if ($("#AnalyseProcessEvents").length > 0){
        AnalyseProcessEvents(buffer_output.events)
    }
    else if ($("#AnalyseProcessNetflow").length > 0){
        AnalyseProcessNetflow(buffer_output.traffic)
    }
}

function AnalyseProcessEvents(buffer_events){
	// Добавление в таблицу
    if (length in buffer_events) {
        if (buffer_events.length > 0) {
            data_in_graf = []
            for (let record of buffer_events) {
                buffer_output_events.push(record)
                Append_tr_Events(record)
                data_in_graf.push([record.ID, record.loss])
            }
            appendPointInGraphEvents(data_in_graf);
        }
    }
}

function AnalyseProcessNetflow(buffer_traffic){
	// Добавление в таблицу
    if (length in buffer_traffic) {
        if (buffer_traffic.length > 0) {
            data_in_graf = []
            for (let record of buffer_traffic) {
                buffer_output_traffic.push(record)
                Append_tr_NetFlow(record)
                data_in_graf.push([record.ID, record.loss])
            }
            appendPointInGraphNetflow(data_in_graf);
        }
    }
}

function Append_tr_Events(record){
    var row = `<tr `

    if ('anomaly' in record) {
        if (record.anomaly == 1){
            row += `style="background: #FD3050; color: #FFFFFF; font-weight: bold;"`
        }

        row += `><td width="5%">${record.ID}</td>`
        row += `<td width="20%">${timeConverter(record.Time_Stamp_Start)}</td>`
        row += `<td width="25%">${timeConverter(record.Time_Stamp_End)}</td>`
        row += `<td width="25%">${record.Process_Name}</td>`

        var conns = ``
        if ('connection' in record) {
            if (record.connection != 0) {
                for (let i = 0; i < record.connection.length; i++){
                    if (i == 0) {
                        conns += `IP: ${num2string(record.connection[i][0])}`
                        conns += ` Порт: ${record.connection[i][1]}`
                    }
                    else {
                        conns += `<br>IP: ${num2string(record.connection[i][0])}`
                        conns += ` Порт: ${record.connection[i][1]}`
                    }
                }
            }
        }

        row += `<td width="25%">${conns}</td></tr>`

        $('#results').append(row);
    }
    if($('#results tr').length >= 10) {
        $('#records').addClass('add-scroll');
    }
}

function Append_tr_NetFlow(record){
    var row = `<tr `

    if ('anomaly' in record) {
        if (record.anomaly == 1){
            row += `style="background: #FD3050; color: #FFFFFF; font-weight: bold;"`
        }

        row += `><td width="5%">${record.ID}</td>`
        row += `<td width="22.5%">${timeConverter(record.Time_Stamp_Start)}</td>`
        row += `<td width="22.5%">${timeConverter(record.Time_Stamp_End)}</td>`

        var conns_src = `IP: ${num2string(record.Src_IP_Flow)}`
        conns_src    += ` Порт: ${record.Src_Port_Flow}`
        row += `<td width="25%">${conns_src}</td>`

        var conns_dst = `IP: ${num2string(record.Dst_IP_Flow)}`
        conns_dst    += ` Порт: ${record.Dst_Port_Flow}`
        row += `<td width="25%">${conns_dst}</td></tr>`

        $('#results').append(row);
    }
    if($('#results tr').length >= 10) {
        $('#records').addClass('add-scroll');
    }
}

function timeConverter(timestamp){
  var a = new Date((timestamp - EPOCH_AS_FILETIME) / HUNDREDS_OF_NANOSECONDS);
  var months = ['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec'];
  var year = a.getFullYear();
  var month = months[a.getMonth()];
  var date = a.getDate();
  var hour = a.getHours();
  var min = a.getMinutes();
  var sec = a.getSeconds();
  var time = date + ' ' + month + ' ' + year + ' ' + hour + ':' + min + ':' + sec ;
  return time;
}

eel.expose(receiverDataCash);
function receiverDataCash(type_data, data_cash) {
    block = true;
    if (type_data == "data_events") {
        //console.log(data_cash);
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

function num2string(ip) {
    return [24,16,8,0].map(n => (ip >> n) & 0xff).join(".")
}

