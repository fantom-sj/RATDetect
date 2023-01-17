
var widthDiagramEvents = 151

function print_chart(){
    var chart_events = document.getElementById("diagramEvents");

    LineChartEvents = new Chart(chart_events, {
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

function appendPointInGraphEvents(data) {
    //LineChartEvents.data.datasets.pop();
    for (let record of data) {
        label = record[0]
        loss  = record[1]
        LineChartEvents.data.labels.push(label);
        LineChartEvents.data.datasets.forEach((dataset) => {
            dataset.data.push(loss);
        });
        widthDiagramEvents += 1
    }
    LineChartEvents.canvas.parentNode.style.width = `${widthDiagramEvents}vh`
    console.log(LineChartEvents.canvas.parentNode.style.width)
    LineChartEvents.update();
    $('#diagramEventsParent').animate({
        scrollLeft: `+=${widthDiagramEvents * 10}px`
    }, "slow");
}
