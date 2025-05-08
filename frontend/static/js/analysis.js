// document.addEventListener('DOMContentLoaded', function() {
//     fetch('/api/analysis_data') // 请求后端 API 路由
//         .then(response => response.json()) // 将响应转换为 JSON
//         .then(chartData => { // 获取到 JSON 数据后执行以下代码
//             // chartData 现在包含了从后端获取的数据

//     // 创建仪表盘
//     createGauge('gauge1', chartData.accuracy.graphrag, 'purple');
//     createGauge('gauge2', chartData.accuracy.vectorrag, 'blue');
//     createGauge('gauge3', chartData.accuracy.hybridrag, 'green');

//     // 创建饼图
//     createPieChart('pieChart1', Object.values(chartData.errorStatistics.vectorrag));
//     createPieChart('pieChart2', Object.values(chartData.errorStatistics.graphrag));
//     createPieChart('pieChart3', Object.values(chartData.errorStatistics.hybridrag));

//     // 创建雷达图
//     createRadarChart('radarChart1', Object.keys(chartData.evaluationMetrics.vectorrag), Object.values(chartData.evaluationMetrics.vectorrag));
//     createRadarChart('radarChart2', Object.keys(chartData.evaluationMetrics.graphrag), Object.values(chartData.evaluationMetrics.graphrag));
//     createRadarChart('radarChart3', Object.keys(chartData.evaluationMetrics.hybridrag), Object.values(chartData.evaluationMetrics.hybridrag));
// })
// .catch(error => {
//     console.error('Error fetching analysis data:', error);
//     //  可以在这里处理错误，例如显示错误信息在页面上
// });

//     // 创建仪表盘的函数
    function createGauge(elementId, percentage, color) {
        let colorStart, colorStop;
        switch (color) {
            case 'blue':
                colorStart = 'rgba(194, 217, 255, 0.7)'; // Pastel Blue with alpha
                colorStop = 'rgba(77, 119, 255, 0.7)';  // Pastel Blue with alpha
                break;
            case 'green':
                colorStart = 'rgba(176, 242, 180, 0.7)'; // Pastel Green with alpha
                colorStop = 'rgba(50, 205, 50, 0.7)';   // Pastel Green with alpha
                break;
            case 'purple':
                colorStart = 'rgba(224, 176, 255, 0.7)'; // Pastel Purple with alpha
                colorStop = 'rgba(160, 32, 240, 0.7)';  // Pastel Purple with alpha
                break;
            default:
                colorStart = '#A6C8FF';
                colorStop = '#1E4B8B';
        }

        var opts = {
            angle: 0.3, /* **Increased angle value to 0.3 for a wider arc** */
            lineWidth: 0.1,
            radiusScale: 1,
            pointer: {
                length: 0,
                strokeWidth: 0,
                color: '#000000'
            },
            limitMax: false,
            limitMin: false,
            colorStart: colorStart,
            colorStop: colorStop,
            strokeColor: '#E0E0E0',
            generateGradient: true,
            highDpiSupport: true,
            staticZones: [
                {strokeStyle: colorStart, min: 0, max: percentage},
                {strokeStyle: '#E0E0E0', min: percentage, max: 100}
            ],
            staticLabels: {
                font: "12px sans-serif",
                labels: [0, 50, 100],
                color: "#000000",
                fractionDigits: 0
            },
            renderTicks: {
                divisions: 5,
                divWidth: 1.1,
                divLength: 0.7,
                divColor: '#333333',
                subDivisions: 3,
                subLength: 0.5,
                subWidth: 0.6,
                subColor: '#666666'
            }
        };

        var target = document.getElementById(elementId);
        var gauge = new Gauge(target).setOptions(opts);
        gauge.maxValue = 100;
        gauge.setMinValue(0);
        gauge.animationSpeed = 32;
        gauge.value = percentage;
        const percentageElement = document.getElementById('percentage' + elementId);
        if (percentageElement) {
            percentageElement.textContent = percentage + '%';
        }
        return gauge;
    }

//     // 饼图部分
    function createPieChart(canvasId, dataValues) {
        const ctx = document.getElementById(canvasId).getContext('2d');
        return new Chart(ctx, {
            type: 'pie',
            data: {
                labels: ['None Result', 'Lack Information', 'Noisy', 'Other'],
                datasets: [{
                    data: dataValues,
                    backgroundColor: ['#FF6F91', '#00BFFF', '#FFD700', '#00CED1'], /* 更鲜艳的饼图颜色 */
                }],
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                plugins: {
                    legend: {
                        position: 'right',
                        align: 'start'
                    },
                    datalabels: {
                        formatter: (value, context) => {
                            const total = context.dataset.data.reduce((a, b) => a + b, 0);
                            const percentage = (value / total * 100).toFixed(1);
                            return `${percentage}%`;
                        },
                        color: '#fff',
                        font: {
                            weight: 'bold',
                            size: 12
                        },
                    }
                }
            },
            plugins: [ChartDataLabels]
        });
    }


//     // 雷达图部分
//     function createRadarChart(canvasId, labels, dataValues) {
//         const ctx = document.getElementById(canvasId).getContext('2d');
//         return new Chart(ctx, {
//             type: 'radar',
//             data: {
//                 labels: labels,
//                 datasets: [{
//                     label: 'Evaluation Metric',
//                     data: dataValues,
//                     backgroundColor: 'rgba(34, 202, 236, 0.05)', /* 更浅的背景透明度 */
//                     borderColor: 'rgba(34, 202, 236, 0.8)', /* 略微加深雷达图线条颜色 */
//                     borderWidth: 1.2 /* 进一步减细雷达图线条 */
//                 }]
//             },
//             options: {
//                 responsive: true,
//                 maintainAspectRatio: false,
//                 scales: {
//                     r: {
//                         beginAtZero: true,
//                         max: 1,
//                         ticks: {
//                             stepSize: 0.2
//                         },
//                         grid: {
//                             color: 'rgba(0, 0, 0, 0.08)' /* 更淡的网格线 */
//                         },
//                         angleLines: {
//                             color: 'rgba(0, 0, 0, 0.08)' /* 更淡的角度线 */
//                         }
//                     }
//                 },
//                  plugins: {
//                     legend: {
//                         display: false
//                     }
//                 },
//                 elements: {
//                     line: {
//                         tension: 0.1
//                     },
//                     point: {
//                         radius: 1.8 /* 雷达图顶点更小 */
//                     }
//                 }
//             }
//         });
//     }
// });


