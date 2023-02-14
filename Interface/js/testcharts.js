var animation = {
  onProgress: function() {
    const ctx = this.ctx;
    ctx.textAlign = 'center';
    ctx.textBaseline = 'bottom';

    let dataSum = 0;
    var label = undefined;
    if(this._sortedMetasets.length > 0 && this._sortedMetasets[0].data.length > 0) {
      const dataset = this._sortedMetasets[0].data[0].$context.dataset;
      dataSum = dataset.data.reduce((p, c) => p + c, 0);
      label = dataset.label
    }
    if(dataSum <= 0) return;

    this._sortedMetasets.forEach(meta => {
      meta.data.forEach(metaData => {
        const dataset = metaData.$context.dataset;
        const datasetIndex = metaData.$context.dataIndex;

        const value = dataset.data[datasetIndex];
        const mid_radius = metaData.innerRadius + (metaData.outerRadius - metaData.innerRadius) * 0.5;
        const start_angle = metaData.startAngle;
        const end_angle = metaData.endAngle;
        if(start_angle === end_angle) return; // hidden
        const mid_angle = start_angle + (end_angle - start_angle) / 2;

        const x = mid_radius * Math.cos(mid_angle);
        const y = mid_radius * Math.sin(mid_angle);

        ctx.fillStyle = '#000';
        ctx.font = "20px Verdana";
        ctx.fillText(value, metaData.x + x + 5, metaData.y + y + 15);
        ctx.fillText(label, metaData.x, metaData.y + 10);
      });
    });
  }
}

var configAnomalyProcessRatio = {
  type: 'doughnut',
  data: {
    labels: ['Обнаружено', 'Ошибки I рода'],
    datasets: [{
        label: "RAT-троянов",
        data: [0, 0],
        backgroundColor: ["#ffc12e", "#f04037"],
    }]
  },
  options: {
    responsive: true,
    plugins: {
        legend: {
            position: 'top',
        },
        datalabels: {
            display: true,
            align: 'center',
            anchor: 'center'
        },
    },
    cutout: "60%",
    animation: animation
  },
};

var configNormalProcessRatio = {
  type: 'doughnut',
  data: {
    labels: ['Обнаружено', 'Ошибки II рода'],
    datasets: [{
        label: "Легальное ПО",
        data: [0, 0],
        backgroundColor: ["#34f351", "#f04037"],
    }]
  },
  options: {
    responsive: true,
    plugins: {
        legend: {
            position: 'top',
        },
        datalabels: {
            display: true,
            align: 'center',
            anchor: 'center'
        },
    },
    cutout: "60%",
    animation: animation
  },
};

var configAnomalyProcess = {
  type: 'doughnut',
  data: {
    labels: ['Обнаружено', 'Ошибки I рода'],
    datasets: [{
        label: "RAT-троянов",
        data: [0, 0],
        backgroundColor: ["#02f8fd", "#f04037"],
    }]
  },
  options: {
    responsive: true,
    plugins: {
        legend: {
            position: 'top',
        },
        datalabels: {
            display: true,
            align: 'center',
            anchor: 'center'
        },
    },
    cutout: "60%",
    animation: animation
  },
};

var configNormalProcess = {
  type: 'doughnut',
  data: {
    labels: ['Обнаружено', 'Ошибки II рода'],
    datasets: [{
        label: "Легальное ПО",
        data: [0, 0],
        backgroundColor: ["#ff00fc", "#f04037"],
    }]
  },
  options: {
    responsive: true,
    plugins: {
        legend: {
            position: 'top',
        },
        datalabels: {
            display: true,
            align: 'center',
            anchor: 'center'
        },
    },
    cutout: "60%",
    animation: animation
  },
};