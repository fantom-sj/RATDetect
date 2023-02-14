var ChartAnomalyProcessRatio = undefined;
var ChartNormalProcessRatio  = undefined;
var ChartAnomalyProcess      = undefined;
var ChartNormalProcess       = undefined;


function printChartTesting() {
    var canvasAnomalyProcessRatio = document.getElementById("AnomalyProcessRatio");
    ChartAnomalyProcessRatio = new Chart(canvasAnomalyProcessRatio, configAnomalyProcessRatio);

    var canvasNormalProcessRatio = document.getElementById("NormalProcessRatio");
    ChartNormalProcessRatio = new Chart(canvasNormalProcessRatio, configNormalProcessRatio);

    var canvasAnomalyProcess = document.getElementById("AnomalyProcess");
    ChartAnomalyProcess = new Chart(canvasAnomalyProcess, configAnomalyProcess);

    var canvasNormalProcess = document.getElementById("NormalProcess");
    ChartNormalProcess = new Chart(canvasNormalProcess, configNormalProcess);

    updateChartTesting();
}

function updateChartTesting() {
    ChartAnomalyProcessRatio.data.datasets.forEach((dataset) => {
        dataset.data.pop();
        dataset.data.pop();
        dataset.data.push(39);
        dataset.data.push(29);
    });
    ChartAnomalyProcessRatio.update();

    ChartNormalProcessRatio.data.datasets.forEach((dataset) => {
        dataset.data.pop();
        dataset.data.pop();
        dataset.data.push(1058);
        dataset.data.push(1);
    });
    ChartNormalProcessRatio.update();

    ChartAnomalyProcess.data.datasets.forEach((dataset) => {
        dataset.data.pop();
        dataset.data.pop();
        dataset.data.push(3);
        dataset.data.push(0);
    });
    ChartAnomalyProcess.update();

    ChartNormalProcess.data.datasets.forEach((dataset) => {
        dataset.data.pop();
        dataset.data.pop();
        dataset.data.push(23);
        dataset.data.push(0);
    });
    ChartNormalProcess.update();
}
