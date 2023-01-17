
var widthDiagramNetflow = 151

function print_chart(){
    var chart_netflow = document.getElementById("diagramNetflow");

    LineChartNetflow = new Chart(chart_netflow, {
        type: 'line',
        data: {
            labels: [],
            datasets: [{
                data: [],
                borderColor: '#FF6384',
                backgroundColor: '#FF6384',
                label: 'Уровень аномалий',
                fill: true
            }]
        },
        options: {
            plugins: {
                filler: {
                    propagate: false,
                    drawTime: "beforeDraw"
                },
                title: {
                    display: true,
                    text: (chart_events) => "График аномальной активности"
                }
            },
            pointBackgroundColor: '#fff',
            radius: 2,
            interaction: {
                intersect: false,
            },
            elements: {
                line: {
                    tension: 0.2,
                }
            },
            scales: {
                y: {
                    min: -0.02,
                    max: 1.01
                }
            },
            maintainAspectRatio: false,
        },
    });
}

function appendPointInGraphNetflow(data) {
    //LineChartEvents.data.datasets.pop();
    for (let record of data) {
        label = record[0]
        loss  = record[1]
        LineChartNetflow.data.labels.push(label);
        LineChartNetflow.data.datasets.forEach((dataset) => {
            dataset.data.push(loss);
        });
        widthDiagramNetflow += 1
    }
    LineChartNetflow.canvas.parentNode.style.width = `${widthDiagramNetflow}vh`
    //console.log(LineChartNetflow.canvas.parentNode.style.width)
    LineChartNetflow.update();
    $('#diagramNetflowParent').animate({
        scrollLeft: `+=${widthDiagramNetflow * 10}px`
    }, "slow");
}


