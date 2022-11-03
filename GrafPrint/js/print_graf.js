function getData(event) {
    var index_col = 17

	var files = event.target.files;
	var file = files[0];

	var reader = new FileReader();
	reader.readAsText(file);
	reader.onload = function(event){
		var csv = event.target.result;

		const col_names = $.csv.toArrays(csv).slice(0, 1);
		console.log(col_names[0][index_col])
		const data = $.csv.toArrays(csv).slice(1);

		arr = []
		index = 0
		for(let el of data){
			el = Number(el[index_col]);
			arr.push([index, el]);
			index++;
		}
		setTimeout(() => {
			print_graff(arr, col_names[0][index_col]);
		}, 2000);
	}
}

function print_graff(arr, col_name) {
	start 		= 1000
	speed 		= 65
	window_size = 50

	var dataSet = anychart.data.set(arr.slice(0, window_size));

	var firstSeriesData = dataSet.mapAs({ x: 0, value: 1 });

	var chart = anychart.area();
	chart.animation(true, 1000);
	chart.padding([20, 20, 200, 20]);
	chart.crosshair().enabled(true).yLabel(false).yStroke(null);
	chart.tooltip().positionMode('point');
	//chart.tooltip().displayMode('union');
	//chart.title('');
	chart.yAxis().title('Интенсивность');
	chart.xAxis().labels().padding(5);

	var firstSeries = chart.area(firstSeriesData);
	firstSeries.color('#DE834D');
	firstSeries.name(col_name);
	firstSeries.hovered().markers().enabled(true).type('square').size(10); // square diamonds
	firstSeries
		.tooltip()
		.position('right')
		.anchor('left-center')
		.offsetX(5)
		.offsetY(5);
	firstSeries.markers(true);

	chart.legend().enabled(true).fontSize(16).padding([0, 0, 10, 0]);
	chart.container('container');
	chart.autoRedraw(true);
	chart.draw();

	itter_arr = []
	new_arr = arr.slice(window_size)
	for (i=0; i<new_arr.length; i++){
		itter_arr.push(start + (speed * i));
	}

	for (let it of itter_arr){
		setTimeout(() => {
			idx = Math.round((it-start)/speed);
			dataSet.remove(0);
			dataSet.append(new_arr[idx]);
		}, it);
	}
}