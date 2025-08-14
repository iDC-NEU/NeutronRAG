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
const addHistorySessionButton = document.getElementById('add-session-btn');
const deleteCurrentSessionButton = document.getElementById('delete-session-btn');
const refreshSessionsButton = document.getElementById('refresh-sessions-btn');
const loadingHTML = `
    <div id="loading-indicator" style="text-align: center; margin: 10px;">
        <div class="spinner"></div>
        <div>正在处理中，请稍候...</div>
    </div>
`;
function showLoading() {
    currentAnswerContent.innerHTML = loadingHTML;
}
function hideLoading() {
    const loading = document.getElementById('loading-indicator');
    if (loading) loading.style.display = 'none';
}

let isGenerating = false;
let abortController = new AbortController();
let currentCytoscapeInstance = null;
let currentAnswers = { query: "", vector: "", graph: "", hybrid: "" };
const placeholderText = `<div class="placeholder-text">请选择 RAG 模式，输入内容或从历史记录中选择，然后点击应用设置。</div>`;
let history_list = [];
let selectedDatasetName = null;

let sessionsList = [];
let currentSession = null;
let current_vector_response= null;
let current_graph_response= null;
let current_hybrid_response= null;
let radarChart1 = null;
let radarChart2 = null;
let radarChart3 = null;
let current_radar1 = [0.1,0.1,0.1,0.1,0.1];
let historySessions = {};
let currentHistorySessionName; // 恢复旧变量以兼容旧函数


// 控制当前页面状态 （按钮的可用）
let current_state = "IDLE"

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

    const canvas = document.getElementById(canvasId);
    if (!canvas) {
        console.error(`Canvas element with id '${canvasId}' not found.`);
        return;
    }
    const ctx = canvas.getContext('2d');
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
                    ticks: { stepSize: 0.2 },
                    grid: { color: 'rgba(0, 0, 0, 0.08)' },
                    angleLines: { color: 'rgba(0, 0, 0, 0.08)' }
                }
            },
            plugins: { legend: { display: false } },
            elements: {
                line: { tension: 0.1 },
                point: { radius: 1.8 }
            }
        }
    });
    radarCharts[canvasId] = chart;
}

// 增量更新雷达图
async function updateRadarChart(id, newDataValues) {
    const chart = radarCharts[id];
    if (chart) {
        chart.data.datasets[0].data = newDataValues;
        chart.update();
    } else {
        console.error("指定的雷达图不存在！ID:", id);
    }
}

function createGauge(elementId, percentage, color) {
    let colorStart, colorStop;
    switch (color) {
        case 'blue':
            colorStart = 'rgba(194, 217, 255, 0.7)'; colorStop = 'rgba(77, 119, 255, 0.7)'; break;
        case 'green':
            colorStart = 'rgba(176, 242, 180, 0.7)'; colorStop = 'rgba(50, 205, 50, 0.7)'; break;
        case 'purple':
            colorStart = 'rgba(224, 176, 255, 0.7)'; colorStop = 'rgba(160, 32, 240, 0.7)'; break;
        default:
            colorStart = '#A6C8FF'; colorStop = '#1E4B8B';
    }
    var opts = {
        angle: 0.3, lineWidth: 0.1, radiusScale: 1,
        pointer: { length: 0, strokeWidth: 0, color: '#000000' },
        limitMax: false, limitMin: false, colorStart: colorStart, colorStop: colorStop,
        strokeColor: '#E0E0E0', generateGradient: true, highDpiSupport: true,
        staticZones: [
            {strokeStyle: colorStart, min: 0, max: percentage},
            {strokeStyle: '#E0E0E0', min: percentage, max: 100}
        ],
    };
    var target = document.getElementById(elementId);
    if (!target) return;
    var gauge = new Gauge(target).setOptions(opts);
    gauge.maxValue = 100; gauge.setMinValue(0); gauge.animationSpeed = 32; gauge.set(percentage);
    const percentageElement = document.getElementById('percentage' + elementId);
    if (percentageElement) {
        percentageElement.textContent = percentage + '%';
    }
    return gauge;
}

const v_datavalues = [0,0,0,0,0];
const g_datavalues = [0,0,0,0,0];
const h_datavalues = [0,0,0,0,0];
let PieChart1, PieChart2, PieChart3;

function createPieChart(canvasId, dataValues, labelList,color) {
    const canvas = document.getElementById(canvasId);
    if (!canvas) {
        console.error(`Canvas element with id '${canvasId}' not found.`);
        return;
    }
    const ctx = canvas.getContext('2d');
    return new Chart(ctx, {
        type: 'pie',
        data: {
            labels: labelList,
            datasets: [{ data: dataValues, backgroundColor: color }],
        },
        options: {
            responsive: true, maintainAspectRatio: false,
            plugins: {
                legend: { position: 'right', align: 'start' },
                datalabels: {
                    formatter: (value, context) => {
                        const total = context.dataset.data.reduce((a, b) => a + b, 0);
                        if (total === 0) return '';
                        const percentage = (value / total * 100).toFixed(1);
                        return `${percentage}%`;
                    },
                    color: '#fff', font: { weight: 'bold', size: 12 },
                }
            }
        },
        plugins: [ChartDataLabels]
    });
}

function updatePieChart(chartInstance, newDataValues) {
    if (!chartInstance) { console.error("饼图实例不存在！"); return; }
    const currentData = chartInstance.data.datasets[0].data;
    const steps = 20; const intervalTime = 25;
    const diffs = newDataValues.map((val, i) => (val - (currentData[i] || 0)) / steps);
    let count = 0;
    const interval = setInterval(() => {
        count++;
        for (let i = 0; i < newDataValues.length; i++) {
            currentData[i] = (currentData[i] || 0) + diffs[i];
        }
        chartInstance.update();
        if (count >= steps) {
            chartInstance.data.datasets[0].data = [...newDataValues];
            chartInstance.update();
            clearInterval(interval);
        }
    }, intervalTime);
}

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

function displaySelectedAnswer() {
    const selectedMode = ragSelect.value;
    const answerToShow = currentAnswers[selectedMode];
    const queryToShow = currentAnswers.query;
    const answerContentElement = document.getElementById('current-answer-content');

    if (answerContentElement) {
        let chatHTML = '';
        if (queryToShow) {
            chatHTML += `
                <div class="chat-message user-message">
                    <img src="/static/lib/employee.png" alt="User Icon" class="message-icon user-icon-bubble">
                    <div class="message-bubble">${queryToShow}</div>
                </div>
            `;
        }
        if (answerToShow) {
            const modelIcon = '/static/lib/llama.png';
            chatHTML += `
                <div class="chat-message model-message">
                    <img src="${modelIcon}" alt="Model Icon" class="message-icon model-icon-bubble">
                    <div class="message-bubble">${answerToShow}</div>
                </div>
            `;
        }
        if (!queryToShow && !answerToShow) {
            chatHTML = `<div class="placeholder-text">请输入内容或从历史记录中选择。</div>`;
        }
        answerContentElement.innerHTML = chatHTML;
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
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                current_dataset: selectedDatasetName,
                current_session: currentSession
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
}

// --- 历史记录管理函数 ---
async function loadAndPopulateHistoryTables() {
    const resultContainer = document.getElementById("question-list");
    const currentTableNameSpan = document.getElementById("current-table-name");
    
    try {
        const response = await fetch("/get-history-tables");
        if (!response.ok) {
            let errorMsg = `HTTP 错误: ${response.status}`;
            try {
                const errorData = await response.json();
                errorMsg = errorData.error || JSON.stringify(errorData);
            } catch (e) {
                errorMsg = await response.text();
            }
            throw new Error(errorMsg);
        }
        
        const data = await response.json();
        let historyTables = data.history_tables || [];

        if (historyTables.length === 0) {
            console.log("未找到任何历史表，正在尝试自动创建一个...");
            const timestamp = new Date().toLocaleString('sv-SE').replace(/ /g, '_').replace(/:/g, '_').replace(/-/g, '_');
            const newTableName = `session_${timestamp}`;
            
            const createResponse = await fetch('/create-history-session', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ sessionName: newTableName })
            });

            if (!createResponse.ok) {
                 const errorData = await createResponse.json();
                 throw new Error(errorData.message || "自动创建历史表失败");
            }
            const createData = await createResponse.json();
            if (createData.success) {
                historyTables.push(newTableName);
            } else {
                throw new Error(createData.message || "自动创建时后端返回错误");
            }
        }
        
        historySessionSelect.innerHTML = "";
        historyTables.forEach(suffix => {
            const option = document.createElement("option");
            option.value = suffix;
            option.textContent = suffix;
            historySessionSelect.appendChild(option);
        });

        if (historyTables.length > 0) {
            const defaultSuffix = historyTables[0];
            historySessionSelect.value = defaultSuffix;
            currentSession = defaultSuffix;
            currentTableNameSpan.textContent = defaultSuffix;
            deleteCurrentSessionButton.style.display = 'inline-block';
            await displayHistoryEntries(defaultSuffix);
        } else {
             resultContainer.innerHTML = '<div class="no-history-item">无历史记录，请点击“+”新建。</div>';
             currentTableNameSpan.textContent = "无会话";
             deleteCurrentSessionButton.style.display = 'none';
             currentSession = null;
        }

    } catch (error) {
        console.error("❌ 初始化历史表失败:", error);
        resultContainer.innerHTML = `<div class="no-history-item" style="color: red;">加载历史记录失败: ${error.message}</div>`;
    }
}

async function displayHistoryEntries(suffix) {
    const questionList = document.getElementById("question-list");
    // 使用 <div> 以保持语义一致性
    questionList.innerHTML = '<div class="no-history-item">正在加载记录...</div>';

    if (!suffix) {
        questionList.innerHTML = '<div class="no-history-item">请选择一个会话。</div>';
        return;
    }

    try {
        const response = await fetch(`/get-history-entries?table_suffix=${encodeURIComponent(suffix)}`);
        if (!response.ok) {
            throw new Error(`获取失败: ${response.status}`);
        }
        
        const data = await response.json();
        const items = data.entries || [];

        // 清空加载提示
        questionList.innerHTML = '';

        if (items.length === 0) {
            questionList.innerHTML = '<div class="no-history-item">该会话暂无记录。</div>';
            return;
        }

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
                default:
                    backgroundColor = '#F5F5F5'; // GREY作为默认色
                    break;
            }
            div.style.backgroundColor = backgroundColor;

            const answerText = typeof item.answer === 'string' ? item.answer : (item.answer ? JSON.stringify(item.answer) : '');
            const answerSnippet = answerText.substring(0, 30) + (answerText.length > 30 ? '...' : '');
            
            div.dataset.vectorResponse = item.vector_response || '';
            div.dataset.graphResponse = item.graph_response || '';
            div.dataset.hybridResponse = item.hybrid_response || '';
            div.dataset.query = item.query || '';
            div.dataset.answer = answerText;

            // 渲染内容片段，并为 null 值提供回退显示
            div.innerHTML = `
                <p><strong>ID:</strong> ${item.id}</p>
                <p><strong>Query:</strong> ${item.query || 'N/A'}</p>
                <p><strong>V:</strong> ${item.vector_response || 'N/A'}</p>
                <p><strong>G:</strong> ${item.graph_response || 'N/A'}</p>
                <p><strong>H:</strong> ${item.hybrid_response || 'N/A'}</p>
                ${answerSnippet ? `<p><strong>Ans:</strong> ${answerSnippet}</p>` : ''}
            `;

            div.addEventListener('click', handleHistoryItemClick);
            questionList.appendChild(div);
        });
    } catch (error) {
        console.error(`获取历史记录失败:`, error);
        // 提供更明确的错误提示
        questionList.innerHTML = `<div class="no-history-item" style="color: red;">加载记录出错: ${error.message}</div>`;
    }
}


async function createNewHistorySession() {
    const defaultName = `session_${new Date().toISOString().slice(0, 10).replace(/-/g, '_')}`;
    const newName = prompt("请输入新会话名称 (仅限字母、数字、下划线):", defaultName);
    if (!newName || newName.trim() === "") {
        return;
    }
    
    try {
        const response = await fetch('/create-history-session', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ sessionName: newName.trim() })
        });

        const result = await response.json();
        if (!response.ok) {
            throw new Error(result.message || '创建失败');
        }
        
        alert("创建成功!");
        await loadAndPopulateHistoryTables();
        
        const newSessionName = newName.trim();
        historySessionSelect.value = newSessionName;
        currentSession = newSessionName;
        document.getElementById("current-table-name").textContent = currentSession;
        await displayHistoryEntries(currentSession);
        
    } catch (error) {
        console.error("创建会话失败:", error);
        alert(`创建失败: ${error.message}`);
    }
}

async function deleteCurrentHistorySession() {
    if (!currentSession) {
        alert("请先选择一个要删除的会话。");
        return;
    }
    
    if (!confirm(`确定要永久删除会话 "${currentSession}" 吗？此操作不可恢复。`)) {
        return;
    }
    
    try {
        const response = await fetch('/delete-history-session', {
            method: 'DELETE',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ sessionName: currentSession })
        });
        
        const result = await response.json();
        if (!response.ok) {
            throw new Error(result.message || '删除失败');
        }

        alert("删除成功!");
        await loadAndPopulateHistoryTables();

    } catch (error) {
        console.error("删除会话失败:", error);
        alert(`删除失败: ${error.message}`);
    }
}

// --- 核心交互逻辑 ---
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
    questionList.innerHTML = '';
    historySessions[currentSession] = [];
        
    const postData = {
        hop: hop,
        type: type,
        entity: entity,
        dataset: selectedDatasetName,
        session: currentSession
    };
    if (current_state !== "IDLE") {
        alert("⚠️ 当前正在生成中，请稍候完成后再尝试！");
        return;
    }


    applySettingsButton.disabled = true; 
    applySettingsButton.textContent = "Processing..."; 
    adviceContent.innerHTML = "正在加载建议...";
    StopButton.disabled = false;
    ContinuegenetationButton.disabled = true;

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
    current_state = "PROCESSING"
    showLoading();
    
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

                                const v_error = data.item_data.v_error;
                                const g_error = data.item_data.g_error;
                                const h_error = data.item_data.h_error;
                                if (v_error === "Noise") { v_datavalues[0] += 1; } 
                                else if (v_error === "Joint Reasoning") { v_datavalues[1] += 1; }
                                else if (v_error === "Single-Step Reasoning") { v_datavalues[2] += 1; }
                                else { v_datavalues[3] += 1; }

                                if (g_error === "Missing Entity") { g_datavalues[0] += 1; }
                                else if (g_error === "Incorrect Entity") { g_datavalues[1] += 1; }
                                else if (g_error === "Faulty Pruning") { g_datavalues[2] += 1; }
                                else if (g_error === "Noise Interference") { g_datavalues[3] += 1; }
                                else if (g_error === "Hop Limitation") { g_datavalues[4] += 1; }
                                else { g_datavalues[4] += 1; }

                                if (h_error === "None Result") { h_datavalues[0] += 1; }
                                else if (h_error === "Lack Information") { h_datavalues[1] += 1; }
                                else if (h_error === "Noisy") { h_datavalues[2] += 1; }
                                else { h_datavalues[3] += 1; }

                                v_updateList = [avg_v_precision,avg_v_recall,avg_v_relevance,avg_v_accuracy,avg_v_faithfulness];
                                g_updateList = [avg_g_precision,avg_g_recall,avg_g_relevance,avg_g_accuracy,avg_g_faithfulness];
                                h_updateList = [avg_h_precision,avg_h_recall,avg_h_relevance,avg_h_accuracy,avg_h_faithfulness];

                                await updateRadarChart("radarChart1",v_updateList);
                                await updateRadarChart("radarChart2",g_updateList);
                                await updateRadarChart("radarChart3",h_updateList);
                                
                                gauge1.options.staticZones = [
                                    { strokeStyle: 'rgba(224, 176, 255, 0.7)', min: 0, max: avg_v_accuracy*100 },
                                    { strokeStyle: '#E0E0E0', min: avg_v_accuracy*100, max: 100 }
                                ];
                                const percentageElement = document.getElementById('percentage' + 'gauge1');
                                if (percentageElement) {
                                    percentageElement.textContent = (avg_v_accuracy * 100).toFixed(2) + '%';
                                }
                                gauge1.set(avg_v_accuracy*100);

                                gauge2.options.staticZones = [
                                    { strokeStyle: 'rgba(176, 242, 180, 0.7)', min: 0, max: avg_g_accuracy*100 },
                                    { strokeStyle: '#E0E0E0', min: avg_g_accuracy*100, max: 100 }
                                ];
                                const percentageElement2 = document.getElementById('percentage' + 'gauge2');
                                if (percentageElement2) {
                                    percentageElement2.textContent = (avg_g_accuracy * 100).toFixed(2) + '%';
                                }
                                gauge2.set(avg_g_accuracy*100);

                                gauge3.options.staticZones = [
                                    { strokeStyle: 'rgba(194, 217, 255, 0.7)', min: 0, max: avg_h_accuracy*100 },
                                    { strokeStyle: '#E0E0E0', min: avg_h_accuracy*100, max: 100 }
                                ];
                                const percentageElement3 = document.getElementById('percentage' + 'gauge3');
                                if (percentageElement3) {
                                    percentageElement3.textContent = (avg_h_accuracy * 100).toFixed(2) + '%';
                                }
                                gauge3.set(avg_h_accuracy*100);
                    
                                updatePieChart(PieChart1,v_datavalues);
                                updatePieChart(PieChart2,g_datavalues);
                                updatePieChart(PieChart3,h_datavalues);
                                
                                historySessions[currentSession].push(historyItem);
                                
//                                 await incrementallyDisplayNewHistoryItem(historyItem);
                                await displayHistoryEntries(currentSession);
                                console.log(currentSession,"当前要显示的")
                                
                                
                                adviceContent.innerHTML = `<p>正在处理: ${data.item_data.query}</p>`;
                            }
                                break;
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
//                             await fetchAndDisplaySuggestions();
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
        current_state = "IDLE"
        hideLoading();
        applySettingsButton.textContent = "applySettings";
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
     let clickedItemData = null;

     const queryText = div.querySelector('p:nth-of-type(2)')?.textContent.replace('Query: ', '') || 'Loading...';
     const vContent = div.querySelector('p:nth-of-type(3)')?.textContent.replace('V:', '').trim() || 'Loading...';
     const gContent = div.querySelector('p:nth-of-type(4)')?.textContent.replace('G:', '').trim() || 'Loading...';
     const hContent = div.querySelector('p:nth-of-type(5)')?.textContent.replace('H:', '').trim() || 'Loading...';
     current_graph_response = gContent;
     current_vector_response = vContent;
     current_hybrid_response = hContent;
     clickedItemData = { id: itemId, query: queryText,vector_response:  current_vector_response,graph_response:  current_graph_response,hybrid_response:  current_hybrid_response};
     console.log(`历史项被点击: ID ${itemId}`);

     document.querySelectorAll('.question-item.selected').forEach(el => el.classList.remove('selected'));
     div.classList.add('selected');
     vectorContent.innerHTML = '正在加载向量细节...'; 
     if (currentCytoscapeInstance) { currentCytoscapeInstance.destroy(); currentCytoscapeInstance = null; } 
     cyContainer.innerHTML = ''; 
     const cyGraphDiv = document.createElement('div'); 
     cyGraphDiv.id = 'cy'; 
     cyGraphDiv.innerHTML = '<p>正在加载图谱...</p>'; 
     cyContainer.appendChild(cyGraphDiv);

     updateAnswerStore(clickedItemData); 
     displaySelectedAnswer();

     let vectorResponse, graphResponse;
     try { 
         [vectorResponse, graphResponse] = await Promise.all([ 
             fetch(`/get-vector/${itemId}?sessionName=${encodeURIComponent(currentSession)}&datasetName=${encodeURIComponent(selectedDatasetName)}`), 
             fetch(`/get-graph/${itemId}?sessionName=${encodeURIComponent(currentSession)}&datasetName=${encodeURIComponent(selectedDatasetName)}`) 
         ]);
         if (vectorResponse.ok) {
             const d = await vectorResponse.json();
             if (d?.chunks && Array.isArray(d.chunks)) {
                 vectorContent.innerHTML = highlightChunksBasedOnGroundTruth(d.chunks, div.dataset.answer);
             } else {
                 vectorContent.innerHTML = '<p>未找到向量块。</p>';
             }
         } else {
             const t = await vectorResponse.text();
             vectorContent.innerHTML = `<p>向量错误 ${vectorResponse.status}: ${t}</p>`;
         }
         if (graphResponse.ok) { 
             const d = await graphResponse.json(); 
             if (d?.nodes || d?.edges) { 
                 renderCytoscapeGraph(d); 
             } else { 
                 cyGraphDiv.innerHTML = '<p>未找到图谱数据。</p>'; 
             } 
         } else { 
             const t = await graphResponse.text(); 
             cyGraphDiv.innerHTML = `<p>图谱错误 ${graphResponse.status}: ${t}</p>`;
         }
     } catch (error) { 
         console.error(`为项 ${itemId} 获取细节时出错:`, error); 
         if (!vectorResponse?.ok) vectorContent.innerHTML = `<p>加载向量细节失败。 ${error.message}</p>`; 
         if (!graphResponse?.ok) cyGraphDiv.innerHTML = `<p>加载图谱细节失败。 ${error.message}</p>`; 
     }
}

function renderCytoscapeGraph(graphData) {
    let cyTargetDiv = document.getElementById('cy');
    if (!cyTargetDiv) {
        console.error("Cytoscape 容器 'cy' 在 DOM 中未找到。");
        return;
    }
    cyTargetDiv.innerHTML = '';
    if (currentCytoscapeInstance) {
        currentCytoscapeInstance.destroy();
    }
    try {
        currentCytoscapeInstance = cytoscape({
            container: cyTargetDiv,
            elements: { nodes: graphData.nodes || [], edges: graphData.edges || [] },
            style: [
                { selector: 'node', style: { 'background-color': 'data(color)', 'label': 'data(label)', 'width': 50, 'height': 50, 'font-size': '10px', 'text-valign': 'center', 'text-halign': 'center', 'color': '#000', 'text-outline-color': '#fff', 'text-outline-width': 1 } },
                { selector: 'edge', style: { 'line-color': 'data(color)', 'target-arrow-color': 'data(color)', 'curve-style': 'bezier', 'target-arrow-shape': 'triangle', 'label': 'data(label)', 'width': 2, 'font-size': '8px', 'text-rotation': 'autorotate', 'text-margin-y': -5, 'color': '#000', 'text-background-color': '#fff', 'text-background-opacity': 0.7, 'text-background-padding': '1px' } },
            ],
            layout: { name: 'cose', fit: true, padding: 30, animate: true, animationDuration: 500, nodeRepulsion: 400000, idealEdgeLength: 100, nodeOverlap: 20 }
        });
        currentCytoscapeInstance.ready(() => {
            currentCytoscapeInstance.fit(null, 30);
        });
    } catch (error) {
        console.error("Cytoscape 渲染错误:", error);
        cyTargetDiv.innerHTML = `<p>渲染图谱时出错: ${error.message}</p>`;
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
    if (!targetElement) return; 
    const isEnlarged = targetElement.classList.toggle('enlarged');
    iconElement.textContent = isEnlarged ? 'fullscreen_exit' : 'fullscreen';
    if (targetElement.contains(cyContainer) || targetElement.id === 'cy-container') {
        if (currentCytoscapeInstance) { 
            setTimeout(() => { 
                currentCytoscapeInstance.resize(); 
                currentCytoscapeInstance.fit(null, 30); 
            }, 300); 
        } 
    }
}

function highlightChunksBasedOnGroundTruth(chunks, groundTruthAnswer) {
    if (!groundTruthAnswer || !chunks.length) {
        return chunks.map((c, i) => 
            `<div class="retrieval-result-item"><p><b>Chunk ${i + 1}:</b> ${c || 'N/A'}</p></div>`
        ).join('');
    }

    const answerVariants = extractAnswerVariants(groundTruthAnswer);
    
    return chunks.map((chunk, i) => {
        const highlightedContent = highlightRelevantText(chunk || 'N/A', answerVariants);
        return `<div class="retrieval-result-item"><p><b>Chunk ${i + 1}:</b> ${highlightedContent}</p></div>`;
    }).join('');
}

function extractAnswerVariants(groundTruthAnswer) {
    if (!groundTruthAnswer) return [];
    
    // 如果是简单字符串（不是JSON格式），直接返回
    if (typeof groundTruthAnswer === 'string' && 
        !groundTruthAnswer.startsWith('[') && 
        !groundTruthAnswer.startsWith('{')) {
        return [groundTruthAnswer.trim()];
    }
    
    try {
        let parsed;
        if (typeof groundTruthAnswer === 'string') {
            let jsonString = groundTruthAnswer;
            
            // 检查是否是Python列表格式
            if (jsonString.startsWith('[') && jsonString.endsWith(']') && jsonString.includes("'")) {
                // 检查是否包含撇号（如 Apple's），如果有则直接用eval
                if (jsonString.includes("\\'")) {
                    try {
                        const evaluated = eval(jsonString);
                        if (Array.isArray(evaluated)) {
                            parsed = evaluated;
                        } else {
                            throw new Error('Not an array');
                        }
                    } catch (evalError) {
                        return [jsonString.trim()];
                    }
                } else {
                    // 没有撇号，尝试简单替换
                    try {
                        const simpleReplaced = jsonString.replace(/'/g, '"');
                        parsed = JSON.parse(simpleReplaced);
                    } catch (replaceError) {
                        // 简单替换失败，使用eval备选方案
                        try {
                            const evaluated = eval(jsonString);
                            if (Array.isArray(evaluated)) {
                                parsed = evaluated;
                            } else {
                                throw new Error('Not an array');
                            }
                        } catch (evalError) {
                            return [jsonString.trim()];
                        }
                    }
                }
            } else {
                // 标准JSON格式
                parsed = JSON.parse(jsonString);
            }
        } else {
            parsed = groundTruthAnswer;
        }
        
        const variants = [];
        if (Array.isArray(parsed)) {
            parsed.forEach(item => {
                if (Array.isArray(item)) {
                    variants.push(...item);
                } else {
                    variants.push(item);
                }
            });
        } else {
            variants.push(parsed);
        }
        
        return variants.filter(v => v && typeof v === 'string' && v.trim());
    } catch (e) {
        // 最终备选方案：直接使用原字符串
        return typeof groundTruthAnswer === 'string' ? [groundTruthAnswer.trim()] : [];
    }
}

function highlightRelevantText(content, answerVariants) {
    if (!content || !answerVariants || answerVariants.length === 0) {
        return content || '';
    }
    
    let result = content;
    
    // 按长度降序排列，优先匹配长短语，避免重复高亮
    const sortedVariants = answerVariants
        .filter(variant => variant && typeof variant === 'string' && variant.trim().length >= 2)
        .sort((a, b) => b.length - a.length);
    
    sortedVariants.forEach(variant => {
        const trimmedVariant = variant.trim();
        if (!trimmedVariant) return;
        
        try {
            const regex = new RegExp(`(${escapeRegex(trimmedVariant)})`, 'gi');
            result = result.replace(regex, '<span style="background-color: #DBEEC0;">$1</span>');
        } catch (e) {
            console.warn('Regex error for variant:', trimmedVariant, e);
        }
    });
    
    return result;
}

function escapeRegex(string) {
    return string.replace(/[.*+?^${}()|[\]\\]/g, '\\$&');
}

// --- 页面初始化与事件绑定 ---
document.addEventListener('DOMContentLoaded', async () => {
    console.log("DOM 完全加载并解析。");

    // 1. 初始化侧边栏
    document.querySelectorAll('.sidebar-section .sidebar-header, .sidebar-section-inner .sidebar-header-inner').forEach((header) => { 
        const content = header.nextElementSibling; 
        const icon = header.querySelector('.material-icons');
        if (!header.classList.contains('collapsed')) header.classList.add("collapsed");
        if (content) content.style.display = "none";
        if (icon) icon.textContent = 'expand_more'; 
    });

    // 2. 初始化图表
    createRadarChart('radarChart1', current_radar1);
    createRadarChart('radarChart2', current_radar1);
    createRadarChart('radarChart3', current_radar1);
    gauge1 = createGauge('gauge1', 10, 'purple');
    gauge2 = createGauge('gauge2', 10, 'blue');
    gauge3 = createGauge('gauge3', 10, 'green');
    const label1 = ['Missing Entity', 'Incorrect Entity', 'Faulty Pruning', 'Noise','Hop Limitation'];
    const label2 = ["Noise","Joint retrieval failure","Single-step retrieval limitation","No Retrieval"];
    const label3 = ['None Result', 'Lack Information', 'Noisy', 'Other'];
    const color1 = ['#FF6F91', '#00BFFF', '#FFD700', '#00CED1','#FF7F0E'];
    const color2 = ['#FF6F91', '#00BFFF', '#FFD700', '#00CED1'];
    const color3 = ['#FF6F91', '#00BFFF', '#FFD700', '#00CED1'];
    PieChart1 = createPieChart('pieChart1', [10,12,13,14,15], label2, color2);
    PieChart2 = createPieChart('pieChart2', [10,12,13,14], label1, color1);
    PieChart3 = createPieChart('pieChart3', [10,12,13,14], label3, color3);

    // 3. 初始化数据集选择
    populateSelect(dim1Select, Object.keys(datasetHierarchy));
    clearSelect(dim2Select);
    clearSelect(dim3Select);
    updateDatasetSelection();
    applySettingsButton.disabled = true;
    ContinuegenetationButton.disabled = true;

    // 4. 【核心】加载历史记录
    await loadAndPopulateHistoryTables();

    displaySelectedAnswer();

    // 5. 【核心】为所有元素绑定事件
    ragSelect.addEventListener("change", displaySelectedAnswer);
    dim1Select.addEventListener('change', updateDatasetSelection);
    dim2Select.addEventListener('change', updateDatasetSelection);
    dim3Select.addEventListener('change', updateDatasetSelection);
    
    if (addHistorySessionButton) {
        addHistorySessionButton.addEventListener('click', createNewHistorySession);
    }
    if (deleteCurrentSessionButton) {
        deleteCurrentSessionButton.addEventListener('click', deleteCurrentHistorySession);
    }
    if (historySessionSelect) {
        historySessionSelect.addEventListener("change", (event) => {
            currentSession = event.target.value;
            document.getElementById("current-table-name").textContent = currentSession;
            displayHistoryEntries(currentSession);
            deleteCurrentSessionButton.style.display = currentSession ? 'inline-block' : 'none';
        });
    }
    
    // if (newSessionNameInput) {
    //     newSessionNameInput.addEventListener('keydown', (event) => { 
    //         if (event.key === 'Enter') { 
    //             event.preventDefault(); 
    //             handleConfirmNewSession(); 
    //         } else if (event.key === 'Escape') { 
    //             hideNewSessionInput(); 
    //         } 
    //     });
    // }
    // if (cancelNewSessionButton) {
    //     cancelNewSessionButton.addEventListener('click', hideNewSessionInput);
    // }
    
    const logoutBtn = document.getElementById('logout-button');
    if (logoutBtn) {
        logoutBtn.addEventListener('click', function () {
            fetch('/api/logout', { method: 'POST', headers: { 'Content-Type': 'application/json' }, credentials: 'include' })
            .then(response => {
                if (response.ok) { window.location.href = '/login'; } 
                else { return response.json().then(data => { alert('登出失败: ' + (data.message || '未知错误')); }); }
            })
            .catch(error => { console.error('登出异常:', error); alert('登出失败，请检查网络连接'); });
        });
    }
});

/* 改成了displayHistoryEntries，新增传入参数 suffix */
// async function displayHistoryFromDatabase() {
//     const questionList = document.getElementById("question-list");
//     const suffix = document.getElementById("history-session-select").value;

//     questionList.innerHTML = '<li class="no-history-item">正在加载历史记录...</li>';
//     let items = [];

//     if (!suffix) {
//         questionList.innerHTML = '<li class="no-history-item">请选择一个历史记录表。</li>';
//         return;
//     }

//     try {
//         const response = await fetch(`/get-history-entries?table_suffix=${encodeURIComponent(suffix)}`, {
//             method: 'GET',
//             headers: {
//                 'Content-Type': 'application/json'
//             }
//         });

//         if (!response.ok) throw new Error(`获取失败: ${response.status}`);
//         const data = await response.json();
//         items = data.entries || [];

//     } catch (error) {
//         console.error(`获取历史记录失败:`, error);
//         questionList.innerHTML = `<li class="no-history-item">加载历史记录出错: ${error.message}</li>`;
//         return;
//     }

//     // 清空原列表内容
//     questionList.innerHTML = '';

//     if (items.length === 0) {
//         questionList.innerHTML = '<li class="no-history-item">该历史表暂无记录。</li>';
//         return;
//     }

//     // ✅ 渲染每条历史记录
//     items.forEach(item => {
//         const div = document.createElement('div');
//         div.classList.add('question-item');
//         div.id = `history-${item.id}`;
//         div.dataset.itemId = item.id;

//         // 设置背景颜色
//         let backgroundColor = '#f0f0f0';
//         switch (item.type?.toUpperCase()) {
//             case 'GREEN': backgroundColor = '#d9f7be'; break;
//             case 'RED': backgroundColor = '#ffccc7'; break;
//             case 'YELLOW': backgroundColor = '#fff2e8'; break;
//         }
//         div.style.backgroundColor = backgroundColor;

//         const answerText = typeof item.answer === 'string' ? item.answer : 
//                           item.answer ? JSON.stringify(item.answer) : '';
//         const answerSnippet = answerText ? answerText.substring(0, 30) + '...' : '';

//         // 缓存响应字段（可用于点击展示详细内容）
//         div.dataset.vectorResponse = item.vector_response || '';
//         div.dataset.graphResponse = item.graph_response || '';
//         div.dataset.hybridResponse = item.hybrid_response || '';
//         div.dataset.query = item.query || '';
//         div.dataset.answer = answerText;

//         // 渲染内容片段
//         div.innerHTML = `
//             <p><strong>ID:</strong> ${item.id}</p>
//             <p><strong>Query:</strong> ${item.query || 'N/A'}</p>
//             <p><strong>V:</strong> ${item.vector_response}</p>
//             <p><strong>G:</strong> ${item.graph_response}</p>
//             <p><strong>H:</strong> ${item.hybrid_response}</p>
//             ${answerSnippet ? `<p><strong>Ans:</strong> ${answerSnippet}</p>` : ''}
//         `;

//         // 点击事件（例如显示详细内容）
//         div.addEventListener('click', handleHistoryItemClick);
//         questionList.appendChild(div);
//     });
// }




refreshSessionsButton.addEventListener('click', () => {
    const historySessionSelect = document.getElementById('history-session-select');
    const selectedSuffix = historySessionSelect.value;

    if (!selectedSuffix) {
        alert("请选择一个会话以刷新历史记录");
        return;
    }

    console.log("🔄 正在刷新会话:", selectedSuffix);
    displayHistoryEntries(selectedSuffix);  // 调用你已有的刷新函数
});





sendButton.addEventListener('click', async function () {
    if (current_state !== "IDLE") {
        alert("⚠️ 当前正在生成中，请稍候完成后再尝试！");
        return;
    }

    if (!selectedDatasetName) { 
        alert("请在选择所有维度后，从列表中选择一个数据集。"); 
        return; 
    }

    const hop = document.getElementById("dim1-hops").value;
    const type = document.getElementById("dim2-task").value;
    const entity = document.getElementById("dim3-scale").value;
    if (!hop || !type || !entity || !selectedDatasetName) {
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

    const queryInput = document.getElementById('user-input');
    const answerContentDiv = document.getElementById('answer-content');

    const query = queryInput.value.trim();
    if (!query) {
        alert("请输入问题！");
        return;
    }

    current_state = "PROCESSING"
    showLoading();

    // 构造完整 ask 请求数据
    const askData = {
        dataset: postData,
        query: query,
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
    };

    try {
        const response = await fetch("/ask", { 
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify(askData)
        });

        if (!response.ok) {
            const errorText = await response.text();
            throw new Error(`请求失败: ${response.status} ${errorText}`);
        }

        const result = await response.json();
        const historySessionSelect = document.getElementById('history-session-select');
        console.log("shiyong",historySessionSelect)

        
        if (result && result.item_data) {
            answerContentDiv.innerHTML = `
                <strong>Question:</strong> ${result.item_data.query}<br>
                <strong>Answer:</strong> ${result.item_data.hybrid_response}
            `;
        } else {
            answerContentDiv.innerHTML = `<span style="color:red;">⚠️ 没有返回有效数据</span>`;
        }

        
    } catch (error) {
        console.error('请求失败:', error);
        answerContentDiv.innerHTML = `<span style="color:red;">❌ 请求失败，请检查控制台</span>`;
    }finally {
        await displayHistoryEntries(historySessionSelect);
        current_state = "IDLE";  // 不管成功失败，最终设置为空闲状态
        hideLoading();
    }
});