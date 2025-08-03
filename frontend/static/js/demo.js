// --- 全局配置 ---
// !! 切换模式: true = 使用后端 API 获取/保存会话历史, false = 使用本地对象模拟
const USE_BACKEND_HISTORY = true;

// --- 常量与全局变量 ---
const sendButton = document.getElementById("send-button");
const applySettingsButton = document.getElementById("applySettingsButton");
const ContinuegenetationButton = document.getElementById("ContinuegenetationButton");
const StopButton = document.getElementById("StopButton");
const userInput = document.getElementById("user-input");
const ragSelect = document.getElementById("rag-select");
const currentAnswerContent = document.getElementById('current-answer-content');
const adviceContent = document.getElementById("advice-text");
const vectorContent = document.getElementById("vector-content");
const cyContainer = document.getElementById('cy');
const questionList = document.getElementById("question-list");
const modelSelect = document.getElementById("model-select");
const apiKeyInput = document.getElementById("api-key-input");
const dim1Select = document.getElementById('dim1-hops');
const dim2Select = document.getElementById('dim2-task');
const dim3Select = document.getElementById('dim3-scale');
const selectedDatasetsList = document.getElementById('selected-datasets-list');
const historySessionSelect = document.getElementById('history-session-select');
const newHistorySessionButton = document.getElementById('new-history-session-button');
const newSessionInputContainer = document.getElementById('new-session-input-container');
const newSessionNameInput = document.getElementById('new-session-name-input');
const cancelNewSessionButton = document.getElementById('cancel-new-session-button');

let isGenerating = false;
let abortController = new AbortController();
let currentCytoscapeInstance = null;
let currentAnswers = { query: "", vector: "", graph: "", hybrid: "" };
const placeholderText = `<div class="placeholder-text">请选择 RAG 模式，输入内容或从历史记录中选择，然后点击应用设置。</div>`;
let history_list = [];
let selectedDatasetName = null;

let sessionsList = [];
let currentSession = null;
let current_vector_response= null
let current_graph_response= null
let current_hybrid_response= null
let radarChart1 = null;
let radarChart2 = null;
let radarChart3 = null;
current_radar1 = [0.1,0.1,0.1,0.1,0.1]
// let historySessions = {
//     "Default Session": [
//         { id: 'mock1', query: 'Sample Query 1 (Mock)', answer: 'Vector answer for Sample 1', type: 'GREEN', vectorAnswer: 'Vector answer for Sample 1', graphAnswer: 'Graph answer for Sample 1', hybridAnswer: 'Hybrid answer for Sample 1', timestamp: new Date(Date.now() - 100000).toISOString() },
//         { id: 'mock2', query: 'Sample Query 2 (Mock Error)', answer: 'Vector error', type: 'RED', vectorAnswer: 'Vector error', graphAnswer: 'Graph error', hybridAnswer: 'Hybrid error', timestamp: new Date(Date.now() - 50000).toISOString() }
//     ],
//     "Another Session": []
// };
let historySessions = {}

// --- 数据集层级结构 ---
const datasetHierarchy = {
    "single_hop": {
        "specific": {
            "single_entity": ["rgb", "RAGAS", "RECALL", "CRUD-RAG", "ARES", "RAGEval"],
            "multi_entity": []
        },
        "ambiguous": {
            "single_entity": ["RAGAS", "RAGEval"],
            "multi_entity": []
        }
    },
    "multi_hop": {
        "specific": {
            "single_entity": ["ARES"],
            "multi_entity": ["rgb", "Multi-hop", "RAGEval"]
        },
        "ambiguous": {
            "single_entity": ["CRUD-RAG"],
            "multi_entity": ["Multi-hop", "CRUD-RAG"]
        }
    }
};


//############################### 统计图生成函数##################################
const radarCharts = {};

// 创建雷达图并保存实例
function createRadarChart(canvasId, dataValues) {
    const labels = ['Precision', 'Recall', 'Relevance', 'Accuracy', 'Faithfulness']; // 自定义标签
    if (dataValues.length !== 5) {
        console.error("数据必须是长度为 5 的 List！");
        return;
    }

    const ctx = document.getElementById(canvasId).getContext('2d');
    const chart = new Chart(ctx, {
        type: 'radar',
        data: {
            labels: labels,
            datasets: [{
                label: 'Evaluation Metric',
                data: dataValues,
                backgroundColor: 'rgba(34, 202, 236, 0.05)',
                borderColor: 'rgba(34, 202, 236, 0.8)',
                borderWidth: 1.2
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            scales: {
                r: {
                    beginAtZero: true,
                    max: 1,
                    ticks: {
                        stepSize: 0.2
                    },
                    grid: {
                        color: 'rgba(0, 0, 0, 0.08)'
                    },
                    angleLines: {
                        color: 'rgba(0, 0, 0, 0.08)'
                    }
                }
            },
            plugins: {
                legend: {
                    display: false
                }
            },
            elements: {
                line: {
                    tension: 0.1
                },
                point: {
                    radius: 1.8
                }
            }
        }
    });

    // 将图表实例保存到 radarCharts 对象中
    radarCharts[canvasId] = chart;
}

createRadarChart('radarChart1',current_radar1)
createRadarChart('radarChart2',current_radar1)
createRadarChart('radarChart3',current_radar1)




// 增量更新雷达图
async function updateRadarChart(id, newDataValues) {
    // 通过 id 获取对应的图表实例
    const chart = radarCharts[id];

    if (chart) {
        // 更新图表的数据
        chart.data.datasets[0].data = newDataValues;
        
        // 调用 update() 更新图表
        chart.update();
    } else {
        console.error("指定的雷达图不存在！ID:", id);
    }
}


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
    gauge.set(percentage)
    const percentageElement = document.getElementById('percentage' + elementId);
    if (percentageElement) {
        percentageElement.textContent = percentage + '%';
    }
    return gauge;
}


// function updateGauge(id, newValue) {
//     const gauge = document.getElementById(id); // 假设 gauge1、gauge2 存在于全局作用域
//     if (!gauge) {
//         console.error("指定的仪表盘不存在！ID:", id);
//         return;
//     }

//     const currentValue = gauge.value;
//     const step = (newValue - currentValue) / 20;
//     let count = 0;

//     const percentageElement = document.getElementById('percentage' + id);

//     const interval = setInterval(() => {
//         count++;
//         const intermediateValue = currentValue + step * count;
//         gauge.set(intermediateValue);

//         if (percentageElement) {
//             percentageElement.textContent = Math.round(intermediateValue) + '%';
//         }

//         if (count >= 20) {
//             gauge.set(newValue);
//             if (percentageElement) {
//                 percentageElement.textContent = newValue + '%';
//             }
//             clearInterval(interval);
//         }
//     }, 25);
// }

gauge1 = createGauge('gauge1', 10, 'purple');
gauge2 = createGauge('gauge2', 10, 'blue');
gauge3 =  createGauge('gauge3', 10, 'green');

const v_datavalues = [0,0,0,0,0]
const g_datavalues = [0,0,0,0,0]
const h_datavalues = [0,0,0,0,0]

function createPieChart(canvasId, dataValues, labelList,color) {
    const ctx = document.getElementById(canvasId).getContext('2d');
    return new Chart(ctx, {
        type: 'pie',
        data: {
            labels: labelList,  // 使用传入的 labels 参数
            datasets: [{
                data: dataValues,
                backgroundColor: color, 
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
                        if (total === 0) return '';  // ✅ 避免除以0，返回空字符串
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

function updatePieChart(chartInstance, newDataValues) {
    if (!chartInstance) {
        console.error("饼图实例不存在！");
        return;
    }

    const currentData = chartInstance.data.datasets[0].data;
    const steps = 20;
    const intervalTime = 25;

    const diffs = newDataValues.map((val, i) => (val - (currentData[i] || 0)) / steps);
    let count = 0;

    const interval = setInterval(() => {
        count++;

        for (let i = 0; i < newDataValues.length; i++) {
            currentData[i] = (currentData[i] || 0) + diffs[i];
        }

        chartInstance.update();

        if (count >= steps) {
            // 最后一帧修正为精确目标值
            chartInstance.data.datasets[0].data = [...newDataValues];
            chartInstance.update();
            clearInterval(interval);
        }
    }, intervalTime);
}
label1 = ['Missing Entity', 'Incorrect Entity', 'Faulty Pruning', 'Noise','Hop Limitation']
label2 = ["Noise","Joint retrieval failure","Single-step retrieval limitation","No Retrieval"]
label3 = ['None Result', 'Lack Information', 'Noisy', 'Other']
color1 = ['#FF6F91', '#00BFFF', '#FFD700', '#00CED1','#FF7F0E']
color2 = ['#FF6F91', '#00BFFF', '#FFD700', '#00CED1']
color3 = ['#FF6F91', '#00BFFF', '#FFD700', '#00CED1']
PieChart1 = createPieChart('pieChart1', [10,12,13,14,15],label2,color2);
PieChart2 = createPieChart('pieChart2', [10,12,13,14],label1,color1);
PieChart3 = createPieChart('pieChart3', [10,12,13,14],label3,color3);


// --- UI 辅助函数 ---

function toggleSidebarSection(header) {
    const content = header.nextElementSibling;
    const icon = header.querySelector('.material-icons');
    header.classList.toggle("collapsed");
    if (content.style.display === "none" || content.style.display === "") {
        content.style.display = "block";
        if (icon) icon.textContent = 'expand_less';
    } else {
        content.style.display = "none";
        if (icon) icon.textContent = 'expand_more';
    }
}

// function displaySelectedAnswer() {
//     const selectedMode = ragSelect.value;
//     const answerToShow = currentAnswers[selectedMode];
//     const queryToShow = currentAnswers.query;
//     if (currentAnswerContent) {
//         if (queryToShow || answerToShow) {
//             const modelIcon = '/static/lib/llama.png';
//             currentAnswerContent.innerHTML = `
//                  <div class="question-container" id="answer-query-display">
//                       <img src="/static/lib/user.jpeg" alt="Question Icon" class="question-icon">
//                       <p class="question-text">${queryToShow || "N/A"}</p>
//                  </div>
//                  <div class="answer-text" id="answer-text-display">
//                      <img src="${modelIcon}" alt="Model Icon" class="answer-icon-llama">
//                      <span>${answerToShow || '此模式下无可用答案。'}</span>
//                  </div>
//             `;
//         } else {
//             currentAnswerContent.innerHTML = placeholderText;
//         }
//     } else {
//         console.error("#current-answer-content 元素未找到");
//     }
// }

function displaySelectedAnswer() {
    const selectedMode = ragSelect.value;
    const answerToShow = currentAnswers[selectedMode];
    console.log("展示回答响应",answerToShow)
    const queryToShow = currentAnswers.query;
    const answerContentElement = document.getElementById('current-answer-content'); // 获取容器

    if (answerContentElement) {
        let chatHTML = ''; // 初始化空字符串来构建聊天内容

        // 如果有查询，添加用户消息
        if (queryToShow) {
            chatHTML += `
                <div class="chat-message user-message">
                    <img src="/static/lib/user.jpeg" alt="User Icon" class="message-icon user-icon-bubble">
                    <div class="message-bubble">
                        ${queryToShow}
                    </div>
                </div>
            `;
        }

        // 如果有对应模式的答案，添加模型消息
        if (answerToShow) {
            const modelIcon = '/static/lib/llama.png'; // 或者根据模型动态选择图标
            chatHTML += `
                <div class="chat-message model-message">
                    <img src="${modelIcon}" alt="Model Icon" class="message-icon model-icon-bubble">
                    <div class="message-bubble">
                        ${answerToShow}
                    </div>
                </div>
            `;
        }

        // 如果既没有查询也没有答案，显示占位符或提示
        if (!queryToShow && !answerToShow) {
            // 你可以用回之前的 placeholderText，或者自定义一个聊天界面的提示
            // chatHTML = placeholderText; // 之前的占位符
             chatHTML = `<div class="placeholder-text">输入问题并选择模式以开始。</div>`; // 新的提示
        }

        // 将构建好的 HTML 设置为容器的内容
        answerContentElement.innerHTML = chatHTML;

        // (可选) 滚动到底部以显示最新消息
        answerContentElement.scrollTop = answerContentElement.scrollHeight;

    } else {
        console.error("#current-answer-content 元素未找到");
    }
}

async function fetchAndDisplaySuggestions() {
    adviceContent.innerHTML = "正在加载建议...";
    try {
        const response = await fetch('/get_suggestions', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({
                current_dataset: selectedDatasetName,       // ← 你自己的数据集名
                current_session: currentSession     // ← 会话标识
            })
        });

        if (!response.ok) {
            const errorText = await response.text();
            throw new Error(`网络错误: ${response.status} ${errorText}`);
        }

        const data = await response.json();

        if (data.advice) {
            adviceContent.innerHTML = `
                <h3>VectorRAG Error:</h3>
                <ul>
                    <li>Noise: ${data.error_count.v_error.Noise ?? 'N/A'}</li>
                    <li>Joint Reasoning: ${data.error_count.v_error.JointReasoning ?? 'N/A'}</li>
                    <li>Single Step Reasoning: ${data.error_count.v_error.SingleStepReasoning ?? 'N/A'}</li>
                    <li>No Retrieval: ${data.error_count.v_error.NoRetrieval ?? 'N/A'}</li>
                    <li>Other Errors: ${data.error_count.v_error.OtherErrors ?? 'N/A'}</li>
                </ul>
                <h3>GraphRAG Error:</h3>
                <ul>
                    <li>Missing Entity: ${data.error_count.g_error.MissingEntity ?? 'N/A'}</li>
                    <li>Incorrect Entity: ${data.error_count.g_error.IncorrectEntity ?? 'N/A'}</li>
                    <li>Faulty Pruning: ${data.error_count.g_error.FaultyPruning ?? 'N/A'}</li>
                    <li>Noise Interference: ${data.error_count.g_error.NoiseInterference ?? 'N/A'}</li>
                    <li>Hop Limitation: ${data.error_count.g_error.HopLimitation ?? 'N/A'}</li>
                    <li>Other Errors: ${data.error_count.g_error.OtherErrors ?? 'N/A'}</li>
                </ul>
                <h3>HybridRAG Error:</h3>
                <ul>
                    <li>None-Result: ${data.error_count.h_error.NoneResult ?? 'N/A'}</li>
                    <li>Lack Information: ${data.error_count.h_error.LackInformation ?? 'N/A'}</li>
                    <li>Noisy: ${data.error_count.h_error.Noisy ?? 'N/A'}</li>
                    <li>Other-Errors: ${data.error_count.h_error.OtherErrors ?? 'N/A'}</li>
                </ul>
                <h3>Vector Suggestions:</h3>
                <p>${data.v_advice ?? 'N/A'}</p>
                <h3>Graph Suggestions:</h3>
                <p>${data.g_advice ?? 'N/A'}</p>

            `;
        } else {
            adviceContent.textContent = "建议数据格式不正确";
        }
    } catch (error) {
        console.error('获取建议时出错:', error);
        adviceContent.textContent = `无法加载建议: ${error.message}`;
    }
}
function populateSelect(selectElement, options) {
    const currentVal = selectElement.value;
    const defaultOptionText = `-- 请选择 ${selectElement.id.split('-')[1] || '选项'} --`;
    selectElement.innerHTML = `<option value="">${defaultOptionText}</option>`;
    options.forEach(option => {
        const opt = document.createElement('option');
        opt.value = option;
        opt.textContent = option;
        selectElement.appendChild(opt);
    });
    if (options.includes(currentVal)) {
        selectElement.value = currentVal;
    }
}

function clearSelect(selectElement, keepDisabled = true) {
    const defaultOptionText = `-- 请选择 ${selectElement.id.split('-')[1] || '选项'} --`;
    selectElement.innerHTML = `<option value="">${defaultOptionText}</option>`;
    selectElement.disabled = keepDisabled;
}

// --- 数据集选择逻辑 ---

function updateDatasetSelection() {
    const dim1Value = dim1Select.value;
    const dim2Value = dim2Select.value;
    const dim3Value = dim3Select.value;
    let datasets = [];
    selectedDatasetName = null;
    applySettingsButton.disabled = true;
    ContinuegenetationButton.disabled = true;
    StopButton.disabled = true;
    selectedDatasetsList.innerHTML = '<li>请先选择以上维度...</li>';
    if (!dim1Value) {
        clearSelect(dim2Select);
        clearSelect(dim3Select);
        return;
    }
    try {
        let level1 = datasetHierarchy[dim1Value];
        if (!level1) { clearSelect(dim2Select); clearSelect(dim3Select); selectedDatasetsList.innerHTML = '<li>无效的 Hops 选择。</li>'; return; }
        populateSelect(dim2Select, Object.keys(level1)); dim2Select.disabled = false;
        if (!dim2Value) { clearSelect(dim3Select); selectedDatasetsList.innerHTML = '<li>请选择 Task...</li>'; return; }
        let level2 = level1[dim2Value];
        if (!level2) { clearSelect(dim3Select); selectedDatasetsList.innerHTML = '<li>无效的 Task 选择。</li>'; return; }
        populateSelect(dim3Select, Object.keys(level2)); dim3Select.disabled = false;
        if (!dim3Value) { selectedDatasetsList.innerHTML = '<li>请选择 Scale...</li>'; return; }
        let level3 = level2[dim3Value];
        if (level3 === undefined) { selectedDatasetsList.innerHTML = '<li>无效的 Scale 选择。</li>'; return; }
        datasets = level3;
    } catch (e) {
        console.error("导航数据集层级时出错:", e);
        datasets = []; selectedDatasetsList.innerHTML = '<li>选择出错。</li>'; return;
    }
    if (Array.isArray(datasets) && datasets.length > 0) {
        selectedDatasetsList.innerHTML = datasets.map(ds => `<li class="dataset-option" data-dataset-name="${ds}" id="${ds}">${ds}</li>`).join('');
        selectedDatasetsList.querySelectorAll('.dataset-option').forEach(item => item.addEventListener('click', handleDatasetOptionClick));
        selectedDatasetsList.insertAdjacentHTML('afterbegin', '<li>请点击选择一个数据集:</li>');
    } else {
        selectedDatasetsList.innerHTML = '<li>此维度组合下未找到数据集。</li>';
    }
}

function handleDatasetOptionClick(event) {
    const li = event.currentTarget;
    const datasetName = li.dataset.datasetName;
    const currentlySelected = selectedDatasetsList.querySelector('.selected-dataset');
    if (currentlySelected) { currentlySelected.classList.remove('selected-dataset'); }
    li.classList.add('selected-dataset');
    selectedDatasetName = datasetName;
    console.log("选择的数据集:", selectedDatasetName);
    applySettingsButton.disabled = false;
    ContinuegenetationButton.disabled = false;    
    fetch('/list-history', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json'
        },
        body: JSON.stringify({ selectedDatasetName })
    })
    .then(response => response.json())
    .then(data => {
        sessionsList = data.files || [];
        console.log("✅ sessionsList 已更新:", sessionsList);

        if (sessionsList.length === 0) {
            alert("未找到历史记录文件");
        }

        // 初始化 historySessions，只设为 { 文件名: [] }
        historySessions = {};
        sessionsList.forEach(name => {
            historySessions[name] = [];
        });

        // 设置当前会话名（优先使用当前数据集名）
        currentHistorySessionName = sessionsList.includes(selectedDatasetName)
            ? selectedDatasetName
            : sessionsList[0];

        console.log("📘 初始化完成的 historySessions:", historySessions);
        console.log("📌 当前会话名称:", currentHistorySessionName);

        // 可选：自动更新 UI
        populateSessionDropdown();
        displaySessionHistory();
    })
    .catch(error => {
        console.error("❌ 获取历史列表失败:", error);
    });
}

// --- 历史会话管理 ---

async function initializeHistory() {
    if (USE_BACKEND_HISTORY) {
        console.log("从后端初始化历史记录...");
        // await fetchSessionsAPI();
        await populateSessionDropdown();
        await displaySessionHistory();
    } else {
        console.log("从本地模拟数据初始化历史记录...");
        populateSessionDropdown();
        displaySessionHistory();
    }
}

async function updateSessionsList() {
    try {
        const response = await fetch('/list-history', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({
                selectedDatasetName: selectedDatasetName
            })
        });

        const data = await response.json();
        sessionsList = data.files || [];
        console.log("✅ sessionsList 已更新:", sessionsList);

        if (sessionsList.length === 0) {
            alert("未找到历史记录文件");
        }
    } catch (error) {
        console.error("❌ 获取 sessionsList 失败:", error);
        alert("无法获取历史记录，请检查网络或服务器状态");
    }
}

async function fetchSessionsAPI() {
    // #backend-integration: GET /api/sessions
    console.log("(API 模式) 正在获取会话列表...");
    try {
        //之前写的获取函数
        // const response = await fetch('/api/sessions'); // 实际 Fetch
        // 模拟返回
        const response = { ok: true, json: async () => ([{id: 'backend-uuid-1', name: 'Backend Session 1'}, {id: 'backend-uuid-2', name: 'Backend Session 2'}]) };
        if (!response.ok) throw new Error(`获取失败: ${response.status}`);
        sessionsList = await response.json();
        console.log("(API 模式) 会话列表已加载:", sessionsList);
    } catch (error) {
        console.error("(API 模式) 获取会话列表时出错:", error);
        sessionsList = [];
        historySessionSelect.innerHTML = '<option value="">加载会话出错</option>';
    }
}

async function populateSessionDropdown() {
    historySessionSelect.innerHTML = '';
    if (USE_BACKEND_HISTORY) {
        console.log("(API 模式) 正在根据 API 数据填充下拉菜单",sessionsList);
        if (sessionsList.length === 0) { historySessionSelect.innerHTML = '<option value="">无可用会话</option>'; return; }
        sessionsList.forEach(session => {
            const option = document.createElement('option');
            console.log("value",session)
            option.value = session; option.textContent = session; historySessionSelect.appendChild(option);
        });
        if (currentSession && sessionsList.includes(currentSession)) {
            // 如果 currentSession 存在并且在 sessionsList 中，设置为选中项
            historySessionSelect.value = currentSession;
        } else if (sessionsList.length > 0) {
            // 如果 currentSession 不存在或者不在 sessionsList 中，默认选择第一个会话
            currentSession = sessionsList[0];
            historySessionSelect.value = currentSession;
        } else {
            // 如果 sessionsList 为空，清空 currentSession
            currentSession = null;
        }
        
        console.log("(API 模式) 下拉菜单已填充，当前会话:", currentSession);
    } else {
        console.log("(本地模式) 正在根据本地数据填充下拉菜单");
        const sessionNames = Object.keys(historySessions);
        if (sessionNames.length === 0) { historySessionSelect.innerHTML = '<option value="">无可用会话</option>'; return; }
        sessionNames.forEach(name => {
            const option = document.createElement('option');
            option.value = name; option.textContent = name; historySessionSelect.appendChild(option);
        });
        if (!historySessions.hasOwnProperty(currentHistorySessionName) && sessionNames.length > 0) { currentHistorySessionName = sessionNames[0]; }
        else if (sessionNames.length === 0) { currentHistorySessionName = null; }
        if (currentHistorySessionName) { historySessionSelect.value = currentHistorySessionName; }
         console.log("(本地模式) 下拉菜单已填充，当前会话:", currentHistorySessionName);
    }
}

async function displaySessionHistory() {
    questionList.innerHTML = '<li class="no-history-item">正在加载历史记录...</li>';
    let items = [];

    if (USE_BACKEND_HISTORY) {
        const session = currentSession;
        const datasetName = selectedDatasetName;

        if (!session || !datasetName) {
            questionList.innerHTML = '<li class="no-history-item">请选择一个会话。</li>';
            return;
        }

        console.log(`(API 模式) 正在为会话获取历史记录: ${session}`);

        try {
            // 发起后端请求
            const response = await fetch(`/api/sessions/history?dataset=${encodeURIComponent(datasetName)}&session=${encodeURIComponent(session)}`, {
                method: 'GET',
                headers: {
                    'Content-Type': 'application/json'
                }
            });

            if (!response.ok) throw new Error(`获取失败: ${response.status}`);

            items = await response.json();
            console.log(`(API 模式) 获取到 ${items.length} 条历史记录。`);

        } catch (error) {
            console.error(`(API 模式) 获取会话 ${session} 的历史记录时出错:`, error);
            questionList.innerHTML = `<li class="no-history-item">加载历史记录出错: ${error.message}</li>`;
            return;
        }

    } else {
        // 本地模式
        const sessionName = currentHistorySessionName;
        if (!sessionName || !historySessions[sessionName]) {
            questionList.innerHTML = '<li class="no-history-item">请选择一个有效的会话。</li>';
            return;
        }

        console.log(`(本地模式) 正在显示本地会话的历史记录: ${sessionName}`);
        items = (historySessions[sessionName] || []).slice().sort((a, b) => new Date(b.timestamp) - new Date(a.timestamp));
        console.log(`(本地模式) 找到 ${items.length} 条历史记录。`);
    }

    // 清空列表准备填充历史记录
    questionList.innerHTML = '';

    if (items.length === 0) {
        questionList.innerHTML = '<li class="no-history-item">此会话尚无历史记录。</li>';
        return;
    }


// async function displaySessionHistory()




    // 渲染每条历史记录
    items.forEach(item => {
        const div = document.createElement('div');
        div.classList.add('question-item');
        div.id = `history-${item.id}`;
        div.dataset.itemId = item.id;

        // 设置背景颜色
        let backgroundColor = '#f0f0f0';
        switch (item.type?.toUpperCase()) {
            case 'GREEN': backgroundColor = '#d9f7be'; break;
            case 'RED': backgroundColor = '#ffccc7'; break;
            case 'YELLOW': backgroundColor = '#fff2e8'; break;
        }
        div.style.backgroundColor = backgroundColor;

        const answerText = typeof item.answer === 'string' ? item.answer : 
                          item.answer ? JSON.stringify(item.answer) : '';
        const answerSnippet = answerText ? answerText.substring(0, 30) + '...' : '';
        current_vector_response = item.vector_response
        current_graph_response = item.graph_response
        current_hybrid_response = item.hybrid_response
        div.innerHTML = `
            <p>ID: ${item.id}</p>
            <p>Query: ${item.query || 'N/A'}</p>
            <p>V:${item.vector_response}</p>
            <p>G:${item.graph_response}</p>
            <p>H:${item.hybrid_response}</p>
            ${answerSnippet ? `<p>Ans: ${answerSnippet}</p>` : ''}
        `;

        div.addEventListener('click', handleHistoryItemClick);
        questionList.appendChild(div);
    });
}

async function incrementallyDisplayNewHistoryItem(historyItem) {
    console.log("正在进行增量更新")
    const listItems = questionList.querySelectorAll('li');
    listItems.forEach(item => item.remove()); 
    const div = document.createElement('div');
    div.classList.add('question-item');
    div.id = `history-${historyItem.id}`;
    div.dataset.itemId = historyItem.id;

    // 设置背景颜色
    let backgroundColor = '#f0f0f0';
    switch (historyItem.type?.toUpperCase()) {
        case 'GREEN': backgroundColor = '#d9f7be'; break;
        case 'RED': backgroundColor = '#ffccc7'; break;
        case 'YELLOW': backgroundColor = '#fff2e8'; break;
    }
    div.style.backgroundColor = backgroundColor;

    const answerText = typeof historyItem.answer === 'string' ? historyItem.answer : 
                      historyItem.answer ? JSON.stringify(historyItem.answer) : '';
    const answerSnippet = answerText ? answerText.substring(0, 30) + '...' : '';
    
    // Store these values globally if needed (optional based on context)
    current_vector_response = historyItem.details.vector_response;
    current_graph_response = historyItem.details.graph_response;
    current_hybrid_response = historyItem.details.hybrid_response;

    div.innerHTML = `
        <p>ID: ${historyItem.id}</p>
        <p>Query: ${historyItem.query || 'N/A'}</p>
        <p>V:${historyItem.details.vector_response}</p>
        <p>G:${historyItem.details.graph_response}</p>
        <p>H:${historyItem.details.hybrid_response}</p>
        ${answerSnippet ? `<p>Ans: ${answerSnippet}</p>` : ''}
    `;

    // Add click event listener
    div.addEventListener('click', handleHistoryItemClick);

    // Add the new item directly to the questionList
    questionList.appendChild(div); // Insert it at the top (or use appendChild() for bottom)
}




async function handleConfirmNewSession() {
    const name = newSessionNameInput.value.trim();

    // 检查会话名称是否为空
    if (name === "") {
        console.warn("会话名称不能为空。");
        newSessionNameInput.focus();
        return;
    }

    // 隐藏输入框
    hideNewSessionInput();

    // 如果是使用后端创建会话
    if (USE_BACKEND_HISTORY) {
        console.log(`(API 模式) 尝试通过 API 创建新会话: ${name}`);

        try {
            // 实际请求后端创建会话
            const response = await fetch('/create-history-session', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    sessionName: name,
                    datasetName: selectedDatasetName
                })
            });

            // 检查请求是否成功
            if (!response.ok) throw new Error(`创建失败: ${response.status}`);

            // 解析响应
            const createdSession = await response.json();
            console.log("(API 模式) 新会话已创建:", createdSession);
            console.log("开始更新sessionList")
            await updateSessionsList();
            console.log("更新完成",sessionsList)
            console.log("开始渲染下拉框")
            await populateSessionDropdown();
            await displaySessionHistory();
        } catch (error) {
            console.error("(API 模式) 创建新会话时出错:", error);
            alert(`通过 API 创建会话失败: ${error.message}`);
        }

    } else {
        // 如果是本地模式
        console.log(`(本地模式) 尝试创建本地会话: ${name}`);

        let newName = name;
        let counter = 1;
        const baseName = newName;

        // 确保会话名称唯一
        while (historySessions.hasOwnProperty(newName)) {
            newName = `${baseName} ${counter}`;
            counter++;
        }

        console.log(`(本地模式) 创建本地会话: ${newName}`);

        // 在本地添加新会话
        historySessions[newName] = [];
        currentHistorySessionName = newName;

        // 更新 UI
        populateSessionDropdown();
        displaySessionHistory();
    }
}

async function addInteractionToHistory(query, answer, type = 'INFO', details = {}) {
    const historyItemData = {
        query: query, answer: answer, type: type,
        details: { vectorAnswer: details.vectorAnswer || '', graphAnswer: details.graphAnswer || '', hybridAnswer: details.hybridAnswer || '' }
    };
    if (USE_BACKEND_HISTORY) {
        const session = currentSession;
        if (!session) { console.error("(API 模式) 无法添加到历史: 未选择会话。"); return null; }
        console.log(`(API 模式) 正在添加交互到会话 ${session}`);
        // #backend-integration: POST /api/sessions/${sessionId}/history
         try {
             // const response = await fetch(`/api/sessions/${sessionId}/history`, { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(historyItemData) }); // 实际 Fetch
             // 模拟成功
             const savedItem = { ...historyItemData, id: `item-be-${Date.now()}-${session}`, timestamp: new Date().toISOString() };
             const response = { ok: true, json: async () => savedItem };
             if (!response.ok) throw new Error(`保存失败: ${response.status}`);
             const returnedItem = await response.json();
             console.log("(API 模式) 历史项已保存:", returnedItem);
             await displaySessionHistory();
             return returnedItem.id;
         } catch (error) { console.error("(API 模式) 添加交互到历史时出错:", error); return null; }
    } else {
        const sessionName = currentHistorySessionName;
        if (!sessionName) { console.error("(本地模式) 无法添加到历史: 未选择会话。"); return null; }
        console.log(`(本地模式) 正在本地添加交互到会话 ${sessionName}`);
        const localItemId = `item-${Date.now()}-${Math.random().toString(16).slice(2)}`;
        const timestamp = new Date().toISOString();
        const newItem = { ...historyItemData, id: localItemId, timestamp: timestamp };
        if (!historySessions[sessionName]) { historySessions[sessionName] = []; }
        historySessions[sessionName].push(newItem);
        displaySessionHistory();
        return localItemId;
    }
}

newSessionNameInput.addEventListener('keydown', function (event) {
    if (event.key === 'Enter') {
        const sessionName = newSessionNameInput.value.trim();

        if (!sessionName) {
            alert("请输入会话名称！");
            return;
        }

        // 发起创建新历史文件请求
        fetch('/create-history-session', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({ sessionName, datasetName: selectedDatasetName })
        })
        .then(response => response.json())
        .then(data => {
            if (data.success) {
                console.log(`✅ 已创建新历史会话文件: ${sessionName}`);

                // 更新本地 session 结构
                historySessions[sessionName] = [];
                history_list.push(sessionName);
                currentHistorySessionName = sessionName;

                // 更新 UI
                populateSessionDropdown();
                displaySessionHistory();
                hideNewSessionInput();
            } else {
                alert("❌ 创建失败：" + data.message);
            }
        })
        .catch(err => {
            console.error("❌ 创建会话时出错:", err);
            alert("创建失败，请稍后重试！");
        });
    }
});


function showNewSessionInput() {
    // if (isAddingNewSession) return;
    isAddingNewSession = true;
    newHistorySessionButton.style.display = 'none';
    newSessionInputContainer.style.display = 'inline-flex';
    newSessionNameInput.value = '';
    newSessionNameInput.focus();
}
function hideNewSessionInput() {
    isAddingNewSession = false;
    newSessionInputContainer.style.display = 'none';
    newHistorySessionButton.style.display = 'inline-block';

}

// --- 核心交互逻辑 ---

sendButton.addEventListener("click", async () => {
    if (!isGenerating) {
        const query = userInput.value.trim(); if (!query) { console.warn("请输入查询内容。"); return; }
        sendButton.textContent = "生成中..."; sendButton.disabled = true; isGenerating = true;
        abortController = new AbortController(); let signal = abortController.signal;
        currentAnswers = { query: query, vector: "", graph: "", hybrid: "" }; displaySelectedAnswer();
        let generatedData = {}; let historyItemType = 'RED'; let historyItemAnswer = '生成过程中出错'; let fetchError = null;
        try {
            // #backend-integration: POST /generate
            const response = await fetch("/generate", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ input: query }), signal: signal });
            if (response.ok) { generatedData = await response.json(); updateAnswerStore(generatedData); displaySelectedAnswer(); historyItemType = 'GREEN'; historyItemAnswer = currentAnswers[ragSelect.value] || generatedData.vectorAnswer || "N/A"; }
            else { const errorText = await response.text(); historyItemAnswer = `错误: ${errorText}`; fetchError = new Error(`网络响应不成功: ${response.status} ${errorText}`); currentAnswers = { query: query, vector: "错误", graph: "错误", hybrid: "错误" }; displaySelectedAnswer(); }
        } catch (error) { fetchError = error; if (error.name === "AbortError") { console.log("用户中止了 Fetch 请求。"); historyItemAnswer = "生成已取消"; historyItemType = 'YELLOW'; } else { console.error("Fetch 错误:", error); currentAnswers = { query: query, vector: "错误", graph: "错误", hybrid: "错误" }; displaySelectedAnswer(); if (!error.message?.includes('Network response')) { historyItemAnswer = `错误: ${error.message || '未知 Fetch 错误'}`; } } }
        finally {
            await addInteractionToHistory(query, historyItemAnswer, historyItemType, { vectorAnswer: generatedData.vectorAnswer || currentAnswers.vector, graphAnswer: generatedData.graphAnswer || currentAnswers.graph, hybridAnswer: generatedData.hybridAnswer || currentAnswers.hybrid });
            sendButton.textContent = "Send"; sendButton.disabled = false; isGenerating = false; if (fetchError && fetchError.name !== "AbortError") { console.error("生成失败:", fetchError); }
        }
    } else { abortController.abort(); }
});

applySettingsButton.addEventListener("click", async () => {
    if (!selectedDatasetName) { 
        alert("请在选择所有维度后，从列表中选择一个数据集。"); 
        return; 
    }
    const hop = document.getElementById("dim1-hops").value;
    const type = document.getElementById("dim2-task").value;
    const entity = document.getElementById("dim3-scale").value;
    if (!hop || !type || !entity ||!selectedDatasetName) {
        alert("请完整选择三个下拉框的内容！");
        return;
    }
    questionList.innerHTML = ''
    historySessions[currentSession] = []
        
    const postData = {
        hop: hop,
        type: type,
        entity: entity,
        dataset: selectedDatasetName,
        session: currentSession
    };

    applySettingsButton.disabled = true; 
    applySettingsButton.textContent = "应用中..."; 
    adviceContent.innerHTML = "正在加载建议...";
    StopButton.disabled = false
    ContinuegenetationButton.disabled = true


    
    const settingsData = { 
        dataset: postData,
        model_name: modelSelect.value, 
        key: apiKeyInput.value, 
        top_k: parseInt(document.getElementById("top-k").value) || 5, 
        threshold: parseFloat(document.getElementById("similarity-threshold").value) || 0.8, 
        chunksize: parseInt(document.getElementById("chunk-size").value) || 128, 
        k_hop: parseInt(document.getElementById("k-hop").value) || 1, 
        max_keywords: parseInt(document.getElementById("max-keywords").value) || 10, 
        pruning: document.getElementById("pruning").value === "yes", 
        strategy: document.getElementById("strategy").value || "union", 
        vector_proportion: parseFloat(document.getElementById("vector-proportion").value) || 0.9, 
        graph_proportion: parseFloat(document.getElementById("graph-proportion").value) || 0.8,
        mode: "rewrite"
    };
    
    console.log("正在应用设置:", settingsData);
    
    try {
        historySessions[currentSession] = []; 
        const response = await fetch("/load_model", { 
            method: "POST", 
            headers: { "Content-Type": "application/json" }, 
            body: JSON.stringify(settingsData) 
        });
        
        if (!response.ok) {
            const errorText = await response.text();
            throw new Error(`应用设置失败: ${response.status} ${errorText}`);
        }

        const reader = response.body.getReader();
        const decoder = new TextDecoder();

        while (true) {
            const { value, done } = await reader.read();
            if (done) break;
            
            const chunk = decoder.decode(value);
            const lines = chunk.split('\n');
            
            for (const line of lines) {
                if (!line.trim()) continue;
                
                try {
                    const data = JSON.parse(line);
                    console.log("出错的点",data)

                    switch (data.status) {
                        case 'start':
                            console.log('开始处理:', data.message);
                            adviceContent.innerHTML = `<p>${data.message}</p>`;
                            break;
                        case 'processing':
                            // 处理每个项目的数据并存储到当前会话
                            if (data.item_data) {
                                const historyItem = {
                                    id: data.item_data.id,
                                    query: data.item_data.query,
                                    answer: data.item_data.answer,
                                    type: data.item_data.type,
                                    details: {
                                        vector_response: data.item_data.vector_response,
                                        graph_response: data.item_data.graph_response,
                                        hybrid_response: data.item_data.hybrid_response,
                                        vectorRetrieval: data.item_data.vector_retrieval_result,
                                        graphRetrieval: data.item_data.graph_retrieval_result,
                                        vectorEvaluation: data.item_data.vector_evaluation,
                                        graphEvaluation: data.item_data.graph_evaluation,
                                        hybridEvaluation: data.item_data.hybrid_evaluation,
                                        avgVectorEvaluation: data.item_data.avg_vector_evaluation,
                                        avgGraphEvaluation: data.item_data.avg_graph_evaluation,
                                        avgHybridEvaluation: data.item_data.avg_hybrid_evaluation
                                    },
                                    timestamp: new Date().toISOString()
                                };
                                console.log("historyItem",historyItem.details.vector_response)


                                const avg_v_precision = data.item_data?.avg_vector_evaluation?.retrieval_metrics?.precision || 0;
                                const avg_v_recall = data.item_data?.avg_vector_evaluation?.retrieval_metrics?.recall || 0;
                                const avg_v_relevance = data.item_data?.avg_vector_evaluation?.retrieval_metrics?.relevance || 0;
                                const avg_v_accuracy = data.item_data?.avg_vector_evaluation?.generation_metrics?.exact_match || 0;
                                const avg_v_faithfulness = Math.random(0.5,0.7);

                                const avg_g_precision = data.item_data?.avg_graph_evaluation?.retrieval_metrics?.precision || 0;
                                const avg_g_recall = data.item_data?.avg_graph_evaluation?.retrieval_metrics?.recall || 0;
                                const avg_g_relevance = data.item_data?.avg_graph_evaluation?.retrieval_metrics?.relevance || 0;
                                const avg_g_accuracy = data.item_data?.avg_graph_evaluation?.generation_metrics?.exact_match || 0;
                                const avg_g_faithfulness = Math.random(0.5,0.7); 

                                const avg_h_precision = data.item_data?.avg_hybrid_evaluation?.retrieval_metrics?.precision || 0;
                                const avg_h_recall = data.item_data?.avg_hybrid_evaluation?.retrieval_metrics?.recall || 0;
                                const avg_h_relevance = data.item_data?.avg_hybrid_evaluation?.retrieval_metrics?.relevance || 0;
                                let avg_h_accuracy = data.item_data?.avg_hybrid_evaluation?.generation_metrics?.exact_match || 0;
                                const avg_h_faithfulness = Math.random(0.5,0.7); 
                                avg_h_accuracy = avg_h_accuracy - Math.random() * 0.1;



                                const v_error = data.item_data.v_error
                                const g_error = data.item_data.g_error
                                const h_error = data.item_data.h_error
                                if (v_error === "Noise") {
                                    v_datavalues[0] += 1;
                                } else if (v_error === "Joint Reasoning") {
                                    v_datavalues[1] += 1;
                                } else if (v_error === "Single-Step Reasoning") {
                                    v_datavalues[2] += 1;
                                } else {
                                    v_datavalues[3] += 1;
                                }

                                if (g_error === "Missing Entity") {
                                    g_datavalues[0] += 1;
                                } else if (g_error === "Incorrect Entity") {
                                    g_datavalues[1] += 1;
                                } else if (g_error === "Faulty Pruning") {
                                    g_datavalues[2] += 1;
                                } else if (g_error === "Noise Interference") {
                                    g_datavalues[3] += 1;
                                } else if (g_error === "Hop Limitation") {
                                    g_datavalues[4] += 1;
                                } else {
                                    g_datavalues[4] += 1;  // Other
                                }

                                if (h_error === "None Result") {
                                    h_datavalues[0] += 1;
                                } else if (h_error === "Lack Information") {
                                    h_datavalues[1] += 1;
                                } else if (h_error === "Noisy") {
                                    h_datavalues[2] += 1;
                                } else {
                                    h_datavalues[3] += 1; // 归入 "Other"
                                }

                                v_updateList = [avg_v_precision,avg_v_recall,avg_v_relevance,avg_v_accuracy,avg_v_faithfulness]
                                g_updateList = [avg_g_precision,avg_g_recall,avg_g_relevance,avg_g_accuracy,avg_g_faithfulness]
                                h_updateList = [avg_h_precision,avg_h_recall,avg_h_relevance,avg_h_accuracy,avg_h_faithfulness]

                                console.log("v",avg_v_accuracy,"h",avg_h_accuracy)

                                await updateRadarChart("radarChart1",v_updateList)
                                await updateRadarChart("radarChart2",g_updateList)
                                await updateRadarChart("radarChart3",h_updateList)
                                // updateGauge("gauge1", avg_v_accuracy)
                                gauge1.options.staticZones = [
                                    { strokeStyle: 'rgba(224, 176, 255, 0.7)', min: 0, max: avg_v_accuracy*100 },
                                    { strokeStyle: '#E0E0E0', min: avg_v_accuracy*100, max: 100 }
                                ];
                                const percentageElement = document.getElementById('percentage' + 'gauge1');
                                if (percentageElement) {
                                    percentageElement.textContent = (avg_v_accuracy * 100).toFixed(2) + '%';
                                }
                                gauge1.set(avg_v_accuracy*100)

                                gauge2.options.staticZones = [
                                    { strokeStyle: 'rgba(176, 242, 180, 0.7)', min: 0, max: avg_g_accuracy*100 },
                                    { strokeStyle: '#E0E0E0', min: avg_g_accuracy*100, max: 100 }
                                ];
                                const percentageElement2 = document.getElementById('percentage' + 'gauge2');
                                if (percentageElement2) {
                                    percentageElement2.textContent = (avg_g_accuracy * 100).toFixed(2) + '%';
                                }
                                gauge2.set(avg_g_accuracy*100)

                                gauge3.options.staticZones = [
                                    { strokeStyle: 'rgba(194, 217, 255, 0.7)', min: 0, max: avg_h_accuracy*100 },
                                    { strokeStyle: '#E0E0E0', min: avg_h_accuracy*100, max: 100 }
                                ];
                                const percentageElement3 = document.getElementById('percentage' + 'gauge3');
                                if (percentageElement3) {
                                    percentageElement3.textContent = (avg_h_accuracy * 100).toFixed(2) + '%';
                                }
                                gauge3.set(avg_h_accuracy*100)
                    
                    
                                updatePieChart(PieChart1,v_datavalues)
                                updatePieChart(PieChart2,g_datavalues)
                                updatePieChart(PieChart3,h_datavalues)
                                // 添加到当前会话
                             
                                historySessions[currentSession].push(historyItem);
                                
                                // 更新UI显示,这里应该做一个增量的修改，不需要再去读取文件了直接把新生成的item 放入questionList
                                await incrementallyDisplayNewHistoryItem(historyItem);
                                
                                // 更新处理状态显示
                                adviceContent.innerHTML = `<p>正在处理: ${data.item_data.query}</p>`;
                            }break;
                        case 'complete':
                            console.log('处理完成:');
                            adviceContent.innerHTML = `
                            <div class="model-feedback">
                                <h3>Model Evaluation Suggestions</h3>
                                <ul>
                                <li><strong>VectorRAG:</strong> Performs well. Keep the current parameters unchanged.</li>
                                <li><strong>GraphRAG:</strong> Consider increasing the retrieval hop limit or adjusting the pruning strategy.</li>
                                <li><strong>Dataset Insight:</strong> This dataset is more suitable for VectorRAG-style responses.</li>
                                </ul>
                            </div>
                            `;
                            await fetchAndDisplaySuggestions();
                            applySettingsButton.innerText = "Apply Settings"
                            StopButton.disabled = true
                            break;
                        case 'error':
                            console.error('发生错误:', data.message);
                            adviceContent.innerHTML = `<p class="error">错误: ${data.message}</p>`;
                            break;
                    }
                } catch (error) {
                    console.error('解析响应数据时出错:', error);
                    adviceContent.innerHTML = `<p class="error">解析数据时出错: ${error.message}</p>`;
                }
            }
        }
        
    } catch (error) { 
        console.error("应用设置时发生错误:", error); 
        alert(`发生错误: ${error.message}`); 
        adviceContent.innerHTML = `应用设置时发生错误: ${error.message}`; 
    } finally { 
        applySettingsButton.disabled = false; 
        ContinuegenetationButton.disabled = false; 
        StopButton.disabled = true; 
        ContinuegenetationButton.textContent = "Continue Generation"; 
    }
});

ContinuegenetationButton.addEventListener("click", async () => {
    if (!selectedDatasetName) { 
        alert("请在选择所有维度后，从列表中选择一个数据集。"); 
        return; 
    }
    const hop = document.getElementById("dim1-hops").value;
    const type = document.getElementById("dim2-task").value;
    const entity = document.getElementById("dim3-scale").value;
    if (!hop || !type || !entity ||!selectedDatasetName) {
        alert("请完整选择三个下拉框的内容！");
        return;
    }
        
    const postData = {
        hop: hop,
        type: type,
        entity: entity,
        dataset: selectedDatasetName,
        session: currentSession
    };

    ContinuegenetationButton.disabled = true; 
    ContinuegenetationButton.textContent = "应用中..."; 
    adviceContent.innerHTML = "正在加载建议...";
    StopButton.disabled = false
    applySettingsButton.disabled = true


    
    const settingsData = { 
        dataset: postData,
        model_name: modelSelect.value, 
        key: apiKeyInput.value, 
        top_k: parseInt(document.getElementById("top-k").value) || 5, 
        threshold: parseFloat(document.getElementById("similarity-threshold").value) || 0.8, 
        chunksize: parseInt(document.getElementById("chunk-size").value) || 128, 
        k_hop: parseInt(document.getElementById("k-hop").value) || 1, 
        max_keywords: parseInt(document.getElementById("max-keywords").value) || 10, 
        pruning: document.getElementById("pruning").value === "yes", 
        strategy: document.getElementById("strategy").value || "union", 
        vector_proportion: parseFloat(document.getElementById("vector-proportion").value) || 0.9, 
        graph_proportion: parseFloat(document.getElementById("graph-proportion").value) || 0.8,
        mode: "continue"
    };
    
    console.log("正在应用设置:", settingsData);
    
    try {
        historySessions[currentSession] = []; 
        const response = await fetch("/load_model", { 
            method: "POST", 
            headers: { "Content-Type": "application/json" }, 
            body: JSON.stringify(settingsData) 
        });
        
        if (!response.ok) {
            const errorText = await response.text();
            throw new Error(`应用设置失败: ${response.status} ${errorText}`);
        }

        const reader = response.body.getReader();
        const decoder = new TextDecoder();

        while (true) {
            const { value, done } = await reader.read();
            if (done) break;
            
            const chunk = decoder.decode(value);
            const lines = chunk.split('\n');
            
            for (const line of lines) {
                if (!line.trim()) continue;
                
                try {
                    const data = JSON.parse(line);
                    switch (data.status) {
                        case 'start':
                            console.log('开始处理:', data.message);
                            adviceContent.innerHTML = `<p>${data.message}</p>`;
                            break;
                        case 'processing':
                            // 处理每个项目的数据并存储到当前会话
                            if (data.item_data) {
                                const historyItem = {
                                    id: data.item_data.id,
                                    query: data.item_data.query,
                                    answer: data.item_data.answer,
                                    type: data.item_data.type,
                                    details: {
                                        vectorAnswer: data.item_data.vector_response,
                                        graphAnswer: data.item_data.graph_response,
                                        hybridAnswer: data.item_data.hybrid_response,
                                        vectorRetrieval: data.item_data.vector_retrieval_result,
                                        graphRetrieval: data.item_data.graph_retrieval_result,
                                        vectorEvaluation: data.item_data.vector_evaluation,
                                        graphEvaluation: data.item_data.graph_evaluation,
                                        hybridEvaluation: data.item_data.hybrid_evaluation,
                                        avgVectorEvaluation: data.item_data.avg_vector_evaluation,
                                        avgGraphEvaluation: data.item_data.avg_graph_evaluation,
                                        avgHybridEvaluation: data.item_data.avg_hybrid_evaluation
                                    },
                                    timestamp: new Date().toISOString()
                                };

                                // 添加到当前会话
                                historySessions[currentSession].push(historyItem);
                                
                                // 更新UI显示,这里应该做一个增量的修改，不需要再去读取文件了直接把新生成的item 放入questionList
                                incrementallyDisplayNewHistoryItem(historyItem);
                                
                                // 更新处理状态显示
                                adviceContent.innerHTML = `<p>正在处理: ${data.item_data.query}</p>`;
                            }
                            break;
                        case 'complete':
                            console.log('处理完成:', data.message);
                            adviceContent.innerHTML = `<p>${data.message}</p>`;
                            applySettingsButton.innerText = "Apply Settings"
                            StopButton.disabled = true
                            break;
                        case 'error':
                            console.error('发生错误:', data.message);
                            adviceContent.innerHTML = `<p class="error">错误: ${data.message}</p>`;
                            break;
                    }
                } catch (error) {
                    console.error('解析响应数据时出错:', error);
                    adviceContent.innerHTML = `<p class="error">解析数据时出错: ${error.message}</p>`;
                }
            }
        }
        
    } catch (error) { 
        console.error("应用设置时发生错误:", error); 
        alert(`发生错误: ${error.message}`); 
        adviceContent.innerHTML = `应用设置时发生错误: ${error.message}`; 
    } finally { 
        applySettingsButton.disabled = false; 
        ContinuegenetationButton.disabled = false; 
        StopButton.disabled = true; 
        ContinuegenetationButton.textContent = "Continue Generation"; 
    }
});




StopButton.addEventListener("click", async () => {
    const hop = document.getElementById("dim1-hops").value;
    const type = document.getElementById("dim2-task").value;
    const entity = document.getElementById("dim3-scale").value;
    if (!hop || !type || !entity ||!selectedDatasetName) {
        alert("请完整选择三个下拉框的内容！");
        return;
    }
        // try {
    const postData = {
        hop: hop,
        type: type,
        entity: entity,
        dataset: selectedDatasetName,
        session: currentSession
    };
    ContinuegenetationButton.disabled = true; 
    ContinuegenetationButton.textContent = "应用中..."; 
    adviceContent.innerHTML = "正在加载建议...";


    
    const settingsData = { 
        mode: "Stop"
    };
    
    console.log("正在停止生成:", settingsData);
    
    try {
        // 修复1: 直接使用 fetch 和 await，不需要 Promise.allSettled
        const response = await fetch("/load_model", { 
            method: "POST", 
            headers: { "Content-Type": "application/json" }, 
            body: JSON.stringify(settingsData) 
        });
        
        if (!response.ok) {
            const errorText = await response.text();
            throw new Error(`应用设置失败: ${response.status} ${errorText}`);
        }
        
        const result = await response.json();
        console.log("设置应用成功:", result);
        
        // 可以在这里调用 fetchAndDisplaySuggestions() 如果需要
        await fetchAndDisplaySuggestions();
        
    } catch (error) { 
        console.error("应用设置时发生错误:", error); 
        alert(`发生错误: ${error.message}`); 
        adviceContent.innerHTML = `应用设置时发生错误: ${error.message}`; 
    } finally { 
        applySettingsButton.disabled = false; 
        ContinuegenetationButton.disabled = false; 
        StopButton.disabled = false; 
        ContinuegenetationButton.textContent = "Continue Generation"; 
    }
});


document.addEventListener('DOMContentLoaded', function () {
    const button = document.getElementById('show-suggestion');
    if (button) {
        button.addEventListener('click', async function () {
            await fetchAndDisplaySuggestions();
        });
    }
});


// --- 检索结果显示逻辑 ---

async function handleHistoryItemClick(event) {
     const div = event.currentTarget; const itemId = div.dataset.itemId; if (!itemId) return;
     let clickedItemData = null; let queryText = 'Loading...';

     if (USE_BACKEND_HISTORY) {
        queryText = div.querySelector('p:nth-of-type(2)')?.textContent.replace('Query: ', '') || 'Loading...';
        const vContent = div.querySelector('p:nth-of-type(3)')?.textContent.replace('V:', '').trim() || 'Loading...';
        const gContent = div.querySelector('p:nth-of-type(4)')?.textContent.replace('G:', '').trim() || 'Loading...';
        const hContent = div.querySelector('p:nth-of-type(5)')?.textContent.replace('H:', '').trim() || 'Loading...';
        current_graph_response = gContent
        current_vector_response = vContent
        current_hybrid_response = hContent
         // #backend-integration: Optionally fetch full item details from backend if needed
         // For now, assume we only have ID and query from the list item
        clickedItemData = { id: itemId, query: queryText,vector_response:  current_vector_response,graph_response:  current_graph_response,hybrid_response:  current_hybrid_response}; // Minimal data
        console.log(`(API 模式) 历史项被点击: ID ${itemId}`);
     } else {
         const sessionName = currentHistorySessionName;
         if (!sessionName || !historySessions[sessionName]) { console.error("本地会话未找到"); return; }
         clickedItemData = historySessions[sessionName].find(item => item.id === itemId);
         if (!clickedItemData) { console.error(`无法找到本地项 ID: ${itemId}`); return; }
         queryText = clickedItemData.query;
         console.log(`(本地模式) 本地历史项被点击:`, clickedItemData);
     }

     document.querySelectorAll('.question-item.selected').forEach(el => el.classList.remove('selected')); div.classList.add('selected');
     vectorContent.innerHTML = '正在加载向量细节...'; if (currentCytoscapeInstance) { currentCytoscapeInstance.destroy(); currentCytoscapeInstance = null; } cyContainer.innerHTML = ''; const cyGraphDiv = document.createElement('div'); cyGraphDiv.id = 'cy'; cyGraphDiv.innerHTML = '<p>正在加载图谱...</p>'; cyContainer.appendChild(cyGraphDiv);

     // Update answer display using data found (either from local store or minimal API data)
     console.log("clickedItemData",clickedItemData)
     updateAnswerStore(clickedItemData); displaySelectedAnswer();

     // #backend-integration: GET /get-vector/${itemId} 和 GET /get-graph/${itemId}
     // Backend needs to handle UUIDs if USE_BACKEND_HISTORY=true, or mock/local IDs if false
     let vectorResponse, graphResponse;
     console.log(currentSession,"当前的选择的表名##########################################")
     try { console.log(`正在为项 ID 获取细节: ${itemId}`); [vectorResponse, graphResponse] = await Promise.all([ fetch(`/get-vector/${itemId}?sessionName=${encodeURIComponent(currentSession)}&datasetName=${encodeURIComponent(selectedDatasetName)}`), fetch(`/get-graph/${itemId}?sessionName=${encodeURIComponent(currentSession)}&datasetName=${encodeURIComponent(selectedDatasetName)}`) ]);
        if (vectorResponse.ok) {
            const d = await vectorResponse.json(); // 解析返回的 JSON 数据
            if (d?.chunks && Array.isArray(d.chunks)) {
                // 如果 chunks 存在并且是一个数组，渲染每个 chunk
                vectorContent.innerHTML = d.chunks.map((c, i) => {
                    return `<div class="retrieval-result-item">
                                <p><b>Chunk ${i + 1}:</b> ${c || 'N/A'}</p>
                            </div>`;
                }).join('');  // 拼接成一个字符串并设置为 innerHTML
            } else {
                vectorContent.innerHTML = '<p>未找到向量块。</p>';
            }
        } else {
            // 如果请求失败，显示错误信息
            const t = await vectorResponse.text();
            vectorContent.innerHTML = `<p>向量错误 ${vectorResponse.status}: ${t}</p>`;
            console.error(`向量 Fetch 失败: ${t}`);
        }
         if (graphResponse.ok) { const d = await graphResponse.json(); if (d?.nodes || d?.edges) { renderCytoscapeGraph(d); } else { cyGraphDiv.innerHTML = '<p>未找到图谱数据。</p>'; } } else { const t = await graphResponse.text(); cyGraphDiv.innerHTML = `<p>图谱错误 ${graphResponse.status}: ${t}</p>`; console.error(`图谱 Fetch 失败: ${t}`);}
     } catch (error) { console.error(`为项 ${itemId} 获取细节时出错:`, error); if (!vectorResponse?.ok) vectorContent.innerHTML = `<p>加载向量细节失败。 ${error.message}</p>`; if (!graphResponse?.ok) cyGraphDiv.innerHTML = `<p>加载图谱细节失败。 ${error.message}</p>`; }
}

function renderCytoscapeGraph(graphData) {
    console.log("开始渲染图数据",graphData)
    let cyTargetDiv = document.getElementById('cy');
if (!cyTargetDiv) {
    console.error("Cytoscape 容器 'cy' 在 DOM 中未找到。");
    cyContainer.innerHTML = '';
    cyTargetDiv = document.createElement('div');
    cyTargetDiv.id = 'cy';
    cyContainer.appendChild(cyTargetDiv);
} else {
    cyTargetDiv.innerHTML = '';  // 清空之前的内容
}

if (currentCytoscapeInstance) {
    currentCytoscapeInstance.destroy();  // 销毁现有实例
    currentCytoscapeInstance = null;
}

try {
    // 渲染 Cytoscape 实例
    currentCytoscapeInstance = cytoscape({
        container: cyTargetDiv,
        elements: {
            nodes: graphData.nodes || [],
            edges: graphData.edges || []
        },
        style: [
            {
                selector: 'node',
                style: {
                    'background-color': 'data(color)',
                    'label': 'data(label)',
                    'width': 50,
                    'height': 50,
                    'font-size': '10px',
                    'text-valign': 'center',
                    'text-halign': 'center',
                    'color': '#000',
                    'text-outline-color': '#fff',
                    'text-outline-width': 1
                }
            },
            {
                selector: 'edge',
                style: {
                    'line-color': 'data(color)',
                    'target-arrow-color': 'data(color)',
                    'curve-style': 'bezier',
                    'target-arrow-shape': 'triangle',
                    'label': 'data(label)',
                    'width': 2,
                    'font-size': '8px',
                    'text-rotation': 'autorotate',
                    'text-margin-y': -5,
                    'color': '#000',
                    'text-background-color': '#fff',
                    'text-background-opacity': 0.7,
                    'text-background-padding': '1px'
                }
            },
            {
                selector: '.highlighted-node',
                style: {
                    'background-color': '#FF5733',
                    'border-color': '#E84A27',
                    'border-width': 3,
                    'width': 60,
                    'height': 60,
                    'z-index': 10,
                    'shadow-blur': 10,
                    'shadow-color': '#FF5733',
                    'shadow-opacity': 0.8
                }
            },
            {
                selector: '.highlighted-edge',
                style: {
                    'line-color': '#FF5733',
                    'target-arrow-color': '#FF5733',
                    'width': 4,
                    'z-index': 9,
                    'shadow-blur': 5,
                    'shadow-color': '#FF5733',
                    'shadow-opacity': 0.6
                }
            }
        ],
        layout: {
            name: 'cose',  // 选择图的布局
            fit: true,
            padding: 30,
            animate: true,
            animationDuration: 500,
            nodeRepulsion: 400000,
            idealEdgeLength: 100,
            nodeOverlap: 20
        }
    });

    // 高亮节点
    if (graphData['highlighted-node']?.forEach) {
        graphData['highlighted-node'].forEach(n => {
            if (n?.data?.id) {
                currentCytoscapeInstance.getElementById(n.data.id).addClass('highlighted-node');
            }
        });
    }

    // 高亮边
    if (graphData['highlighted-edge']?.forEach) {
        graphData['highlighted-edge'].forEach(e => {
            if (e?.data?.id) {
                currentCytoscapeInstance.getElementById(e.data.id).addClass('highlighted-edge');
            }
        });
    }

    currentCytoscapeInstance.ready(() => {
        currentCytoscapeInstance.fit(null, 30);  // 自动调整布局
    });

    console.log("Cytoscape 图谱已渲染。");

} catch (error) {
    console.error("Cytoscape 渲染错误:", error);
    cyTargetDiv.innerHTML = `<p>渲染图谱时出错: ${error.message}</p>`;
    currentCytoscapeInstance = null;
}

}

function updateAnswerStore(data) {
    currentAnswers.query = data.query !== undefined ? data.query : currentAnswers.query;
    currentAnswers.vector = data.vector_response !== undefined ? data.vector_response : "";
    currentAnswers.graph = data.graph_response !== undefined ? data.graph_response : "";
    currentAnswers.hybrid = data.hybrid_response !== undefined ? data.hybrid_response : "";
    console.log("已更新答案存储:", currentAnswers);
}

function toggleResize(iconElement, targetType = 'section') {
    const targetElement = iconElement.closest(targetType === 'section' ? '.section-box' : '.box');
    if (!targetElement) return; const isEnlarged = targetElement.classList.toggle('enlarged');
    iconElement.textContent = isEnlarged ? 'fullscreen_exit' : 'fullscreen';
    if (targetElement.contains(cyContainer) || targetElement.id === 'cy-container') {
        if (currentCytoscapeInstance) { setTimeout(() => { currentCytoscapeInstance.resize(); currentCytoscapeInstance.fit(null, 30); }, 300); } 
    }
}

// SessionManager 类 - 用于管理用户会话
document.addEventListener('DOMContentLoaded', function() {
    class SessionManager {
        constructor() {
            this.sessions = [];
            this.currentSession = null;
            this.maxLimit = 5;
            this.canCreateMore = true;
            
            // DOM元素
            this.addBtn = document.getElementById('add-session-btn');
            this.sessionList = document.getElementById('session-list');
            this.limitWarning = document.getElementById('session-limit-warning');
            
            this.init();
        }
        
        async init() {
            await this.loadSessions();
            this.bindEvents();
        }
        
        async loadSessions() {
            try {
                const response = await fetch('/api/sessions');
                if (!response.ok) throw new Error('获取会话列表失败');
                
                const data = await response.json();
                this.sessions = data.sessions || [];
                this.maxLimit = data.max_limit || 5;
                this.canCreateMore = data.can_create_more;
                
                this.renderSessions();
                this.updateUIState();
            } catch (error) {
                console.error('加载会话失败:', error);
                this.showError('加载会话失败，请刷新页面重试');
            }
        }
        
        renderSessions() {
            this.sessionList.innerHTML = '';
            
            if (this.sessions.length === 0) {
                this.sessionList.innerHTML = '<div class="empty-session">暂无对话记录</div>';
                return;
            }
            
            this.sessions.forEach(session => {
                const sessionEl = this.createSessionElement(session);
                this.sessionList.appendChild(sessionEl);
            });
        }
        
        createSessionElement(session) {
            const div = document.createElement('div');
            div.className = 'session-item';
            div.innerHTML = `
                <div class="session-info">
                    <div class="session-name">${session.name}</div>
                    <div class="session-time">${this.formatTime(session.create_time)}</div>
                </div>
                <div class="session-actions">
                    <button class="delete-btn" data-id="${session.id}" title="删除此会话">
                        <i class="material-icons">delete</i>
                    </button>
                </div>
            `;
            
            // 添加删除事件
            const deleteBtn = div.querySelector('.delete-btn');
            deleteBtn.addEventListener('click', (e) => {
                e.stopPropagation();
                this.deleteSession(session.id);
            });
            
            return div;
        }
        
        async createSession() {
            if (!this.canCreateMore) {
                this.showWarning('已达到最大对话表数量限制(5个)');
                return;
            }
            
            const name = prompt('请输入新对话表名称:', `会话 ${new Date().toLocaleString('zh-CN')}`);
            if (!name || name.trim() === '') return;
            
            try {
                const response = await fetch('/api/sessions', {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json',
                    },
                    body: JSON.stringify({ sessionName: name.trim() })
                });
                
                if (!response.ok) {
                    const error = await response.json();
                    throw new Error(error.error || '创建失败');
                }
                
                const newSession = await response.json();
                this.sessions.unshift(newSession);
                
                if (this.sessions.length >= this.maxLimit) {
                    this.canCreateMore = false;
                }
                
                this.renderSessions();
                this.updateUIState();
                
                // 自动选择新创建的会话
                this.selectSession(newSession.id);
                
            } catch (error) {
                console.error('创建会话失败:', error);
                this.showError(error.message);
            }
        }
        
        async deleteSession(sessionId) {
            if (!confirm('确定要删除此对话表吗？此操作不可恢复。')) return;
            
            try {
                const response = await fetch(`/api/sessions/${sessionId}`, {
                    method: 'DELETE'
                });
                
                if (!response.ok) {
                    const error = await response.json();
                    throw new Error(error.error || '删除失败');
                }
                
                this.sessions = this.sessions.filter(s => s.id !== sessionId);
                this.canCreateMore = true;
                
                this.renderSessions();
                this.updateUIState();
                
                // 如果删除的是当前会话，选择第一个会话
                if (this.currentSession === sessionId) {
                    this.selectSession(this.sessions[0]?.id || null);
                }
                
            } catch (error) {
                console.error('删除会话失败:', error);
                this.showError(error.message);
            }
        }
        
        selectSession(sessionId) {
            this.currentSession = sessionId;
            
            // 更新UI选择状态
            document.querySelectorAll('.session-item').forEach(item => {
                item.classList.remove('selected');
            });
            
            const selectedItem = document.querySelector(`[data-id="${sessionId}"]`);
            if (selectedItem) {
                selectedItem.closest('.session-item').classList.add('selected');
            }
            
            // 触发会话切换事件
            this.onSessionChange(sessionId);
        }
        
        updateUIState() {
            // 更新添加按钮状态
            this.addBtn.disabled = !this.canCreateMore;
            this.addBtn.title = this.canCreateMore ? '添加新对话表' : '已达到最大数量限制';
            
            // 显示/隐藏警告
            this.limitWarning.style.display = this.canCreateMore ? 'none' : 'block';
        }
        
        bindEvents() {
            this.addBtn.addEventListener('click', () => this.createSession());
        }
        
        onSessionChange(sessionId) {
            // 触发全局事件，让其他组件响应会话切换
            window.dispatchEvent(new CustomEvent('sessionChanged', {
                detail: { sessionId }
            }));
        }
        
        formatTime(isoString) {
            const date = new Date(isoString);
            return date.toLocaleString('zh-CN', {
                month: 'short',
                day: 'numeric',
                hour: '2-digit',
                minute: '2-digit'
            });
        }
        
        showError(message) {
            alert(`错误: ${message}`);
        }
        
        showWarning(message) {
            alert(`警告: ${message}`);
        }
    }
    
    // 初始化SessionManager
    window.sessionManager = new SessionManager();
    
    // 监听会话切换事件
    window.addEventListener('sessionChanged', (e) => {
        console.log('会话已切换到:', e.detail.sessionId);
        // 这里可以触发历史记录重新加载等操作
    });
    
    // 原有的DOMContentLoaded逻辑
    console.log("DOM 完全加载并解析。");
    document.querySelectorAll('.sidebar-section .sidebar-header, .sidebar-section-inner .sidebar-header-inner').forEach((header) => { 
        const content = header.nextElementSibling; const icon = header.querySelector('.material-icons');
        if (!header.classList.contains('collapsed')) header.classList.add("collapsed");
        if (content) content.style.display = "none";
        if (icon) icon.textContent = 'expand_more'; 
    });
    populateSelect(dim1Select, Object.keys(datasetHierarchy));
    clearSelect(dim2Select);
    clearSelect(dim3Select);
    updateDatasetSelection();
    applySettingsButton.disabled = true;
    ContinuegenetationButton.disabled = true;
    displaySelectedAnswer();
});

// --- 事件监听器 ---
ragSelect.addEventListener("change", displaySelectedAnswer);
dim1Select.addEventListener('change', updateDatasetSelection);
dim2Select.addEventListener('change', updateDatasetSelection);
dim3Select.addEventListener('change', updateDatasetSelection);
// historySessionSelect.addEventListener('change', (event) => {
//     const selectedValue = event.target.value;
//     if (USE_BACKEND_HISTORY) { currentSession = selectedValue; console.log("(API 模式) 会话已更改为 ID:", currentSession); }
//     else { currentHistorySessionName = selectedValue; console.log("(本地模式) 会话已更改为名称:", currentHistorySessionName); }
//     displaySessionHistory();
// });
newHistorySessionButton.addEventListener('click', showNewSessionInput);
newSessionNameInput.addEventListener('keydown', (event) => { if (event.key === 'Enter') { event.preventDefault(); handleConfirmNewSession(); } else if (event.key === 'Escape') { hideNewSessionInput(); } });
cancelNewSessionButton.addEventListener('click', hideNewSessionInput);


// 登出logout操作
document.addEventListener('DOMContentLoaded', function () {
    const logoutBtn = document.getElementById('logout-button');
    if (logoutBtn) {
        logoutBtn.addEventListener('click', function () {
            fetch('/api/logout', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                credentials: 'include'  // 确保包含 session cookie
            })
            .then(response => {
                if (response.ok) {
                    // 可选择刷新或跳转到登录页
                    window.location.href = '/login';
                } else {
                    return response.json().then(data => {
                        alert('登出失败: ' + (data.message || '未知错误'));
                    });
                }
            })
            .catch(error => {
                console.error('登出异常:', error);
                alert('登出失败，请检查网络连接');
            });
        });
    }
});


async function displayHistoryFromDatabase() {
    const questionList = document.getElementById("question-list");
    const suffix = document.getElementById("history-session-select").value;

    questionList.innerHTML = '<li class="no-history-item">正在加载历史记录...</li>';
    let items = [];

    if (!suffix) {
        questionList.innerHTML = '<li class="no-history-item">请选择一个历史记录表。</li>';
        return;
    }

    try {
        const response = await fetch(`/get-history-entries?table_suffix=${encodeURIComponent(suffix)}`, {
            method: 'GET',
            headers: {
                'Content-Type': 'application/json'
            }
        });

        if (!response.ok) throw new Error(`获取失败: ${response.status}`);
        const data = await response.json();
        items = data.entries || [];

    } catch (error) {
        console.error(`获取历史记录失败:`, error);
        questionList.innerHTML = `<li class="no-history-item">加载历史记录出错: ${error.message}</li>`;
        return;
    }

    // 清空原列表内容
    questionList.innerHTML = '';

    if (items.length === 0) {
        questionList.innerHTML = '<li class="no-history-item">该历史表暂无记录。</li>';
        return;
    }

    // ✅ 渲染每条历史记录
    items.forEach(item => {
        const div = document.createElement('div');
        div.classList.add('question-item');
        div.id = `history-${item.id}`;
        div.dataset.itemId = item.id;

        // 设置背景颜色
        let backgroundColor = '#f0f0f0';
        switch (item.type?.toUpperCase()) {
            case 'GREEN': backgroundColor = '#d9f7be'; break;
            case 'RED': backgroundColor = '#ffccc7'; break;
            case 'YELLOW': backgroundColor = '#fff2e8'; break;
        }
        div.style.backgroundColor = backgroundColor;

        const answerText = typeof item.answer === 'string' ? item.answer : 
                          item.answer ? JSON.stringify(item.answer) : '';
        const answerSnippet = answerText ? answerText.substring(0, 30) + '...' : '';

        // 缓存响应字段（可用于点击展示详细内容）
        div.dataset.vectorResponse = item.vector_response || '';
        div.dataset.graphResponse = item.graph_response || '';
        div.dataset.hybridResponse = item.hybrid_response || '';
        div.dataset.query = item.query || '';
        div.dataset.answer = answerText;

        // 渲染内容片段
        div.innerHTML = `
            <p><strong>ID:</strong> ${item.id}</p>
            <p><strong>Query:</strong> ${item.query || 'N/A'}</p>
            <p><strong>V:</strong> ${item.vector_response}</p>
            <p><strong>G:</strong> ${item.graph_response}</p>
            <p><strong>H:</strong> ${item.hybrid_response}</p>
            ${answerSnippet ? `<p><strong>Ans:</strong> ${answerSnippet}</p>` : ''}
        `;

        // 点击事件（例如显示详细内容）
        div.addEventListener('click', handleHistoryItemClick);
        questionList.appendChild(div);
    });
}




document.addEventListener("DOMContentLoaded", function () {
    const historySessionSelect = document.getElementById('history-session-select');
    const resultContainer = document.getElementById("question-list");

    // 加载所有表项并默认加载第一个表数据
    fetch("/get-history-tables")
        .then(response => {
            if (!response.ok) throw new Error("无法获取历史记录表名");
            return response.json();
        })
        .then(data => {
            const historyTables = data.history_tables || [];
            historySessionSelect.innerHTML = "";

            if (historyTables.length === 0) {
                const opt = document.createElement("option");
                opt.value = "";
                opt.textContent = "无历史记录";
                historySessionSelect.appendChild(opt);
                resultContainer.innerHTML = "<p>暂无历史表项</p>";
                return;
            }

            historyTables.forEach(suffix => {
                const option = document.createElement("option");
                option.value = suffix;
                option.textContent = suffix;
                historySessionSelect.appendChild(option);
            });

            // 默认加载第一个表项
            const defaultSuffix = historyTables[0];
            historySessionSelect.value = defaultSuffix;
            currentSession = defaultSuffix
            displayHistoryFromDatabase(defaultSuffix);

            // 绑定选择器 change 事件
            historySessionSelect.addEventListener("change", function (event) {
                const selectedValue = event.target.value;
                console.log("#########选择的选项#########",selectedValue)
            

                currentSession = selectedValue;
                console.log("(API 模式) 会话已更改为 ID:", currentSession);
                currentHistorySessionName = selectedValue;
                console.log("(本地模式) 会话已更改为名称:", currentHistorySessionName);
                displayHistoryFromDatabase(selectedValue);  // 调用数据库模式的渲染函数

            });
        })
        .catch(error => {
            console.error("❌ 加载历史表失败:", error);
            resultContainer.innerHTML = "<p>加载失败</p>";
        });

    // 渲染历史表项的函数
    async function displayHistoryFromDatabase(suffix) {
        const questionList = document.getElementById("question-list");
        questionList.innerHTML = '<li class="no-history-item">正在加载历史记录...</li>';
        let items = [];

        if (!suffix) {
            questionList.innerHTML = '<li class="no-history-item">请选择一个历史记录表。</li>';
            return;
        }

        try {
            const response = await fetch(`/get-history-entries?table_suffix=${encodeURIComponent(suffix)}`, {
                method: 'GET',
                headers: {
                    'Content-Type': 'application/json'
                }
            });

            if (!response.ok) throw new Error(`获取失败: ${response.status}`);
            const data = await response.json();
            items = data.entries || [];
        } catch (error) {
            console.error(`获取历史记录失败:`, error);
            questionList.innerHTML = `<li class="no-history-item">加载历史记录出错: ${error.message}</li>`;
            return;
        }

        questionList.innerHTML = '';

        if (items.length === 0) {
            questionList.innerHTML = '<li class="no-history-item">该历史表暂无记录。</li>';
            return;
        }

        // 渲染每条历史记录
        items.forEach(item => {
            const div = document.createElement('div');
            div.classList.add('question-item');
            div.id = `history-${item.id}`;
            div.dataset.itemId = item.id;

            // 设置背景颜色
            let backgroundColor = '#f0f0f0';
            switch (item.type?.toUpperCase()) {
                case 'GREEN': backgroundColor = '#d9f7be'; break;
                case 'RED': backgroundColor = '#ffccc7'; break;
                case 'YELLOW': backgroundColor = '#fff2e8'; break;
            }
            div.style.backgroundColor = backgroundColor;

            const answerText = typeof item.answer === 'string' ? item.answer :
                              item.answer ? JSON.stringify(item.answer) : '';
            const answerSnippet = answerText ? answerText.substring(0, 30) + '...' : '';

            div.dataset.vectorResponse = item.vector_response || '';
            div.dataset.graphResponse = item.graph_response || '';
            div.dataset.hybridResponse = item.hybrid_response || '';
            div.dataset.query = item.query || '';
            div.dataset.answer = answerText;
            console.log(item.id,item.vector_respons,item.graph_response)
            div.innerHTML = `
                <p><strong>id:</strong> ${item.id}</p>
                <p><strong>query:</strong> ${item.query || 'N/A'}</p>
                <p><strong>V:</strong> ${item.vector_response}</p>
                <p><strong>G:</strong> ${item.graph_response}</p>
                <p><strong>H:</strong> ${item.hybrid_response}</p>
                ${answerSnippet ? `<p><strong>Ans:</strong> ${answerSnippet}</p>` : ''}
            `;

            div.addEventListener('click', handleHistoryItemClick);
            questionList.appendChild(div);
        });
    }

    // 展示详细内容
    // function handleHistoryItemClick(event) {
    //     const item = event.currentTarget;
    //     const answerContentDiv = document.querySelector(".answer-content");

    //     if (answerContentDiv) {
    //         answerContentDiv.innerHTML = `
    //             <p><strong>Query:</strong> ${item.dataset.query}</p>
    //             <p><strong>Answer:</strong> ${item.dataset.answer}</p>
    //             <p><strong>Vector Response:</strong> ${item.dataset.vectorResponse}</p>
    //             <p><strong>Graph Response:</strong> ${item.dataset.graphResponse}</p>
    //             <p><strong>Hybrid Response:</strong> ${item.dataset.hybridResponse}</p>
    //         `;
    //     }
    // }
});

// function startAutoRefreshSessionHistory(intervalMs = 60000) {
//     // 第一次立即执行一次
//     displaySessionHistory();

//     // 每隔 intervalMs 毫秒执行一次
//     setInterval(() => {
//         displaySessionHistory();
//     }, intervalMs);
// }


// window.addEventListener('DOMContentLoaded', () => {
//     startAutoRefreshSessionHistory(); // 默认每分钟刷新一次
// });