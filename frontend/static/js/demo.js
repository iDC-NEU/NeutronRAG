// --- 全局配置 ---
// !! 切换模式: true = 使用后端 API 获取/保存会话历史, false = 使用本地对象模拟
const USE_BACKEND_HISTORY = false;

// --- 常量与全局变量 ---
const sendButton = document.getElementById("send-button");
const applySettingsButton = document.getElementById("applySettingsButton");
const userInput = document.getElementById("user-input");
const ragSelect = document.getElementById("rag-select");
const currentAnswerContent = document.getElementById('current-answer-content');
const adviceContent = document.getElementById("advice-content");
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
let selectedDatasetName = null;
let isAddingNewSession = false;

// --- 后端模式状态 (仅当 USE_BACKEND_HISTORY = true 时使用) ---
let sessionsList = [];
let currentSessionId = null;

// --- 本地模拟数据 (仅当 USE_BACKEND_HISTORY = false 时使用) ---
let historySessions = {
    "Default Session": [
        { id: 'mock1', query: 'Sample Query 1 (Mock)', answer: 'Vector answer for Sample 1', type: 'GREEN', vectorAnswer: 'Vector answer for Sample 1', graphAnswer: 'Graph answer for Sample 1', hybridAnswer: 'Hybrid answer for Sample 1', timestamp: new Date(Date.now() - 100000).toISOString() },
        { id: 'mock2', query: 'Sample Query 2 (Mock Error)', answer: 'Vector error', type: 'RED', vectorAnswer: 'Vector error', graphAnswer: 'Graph error', hybridAnswer: 'Hybrid error', timestamp: new Date(Date.now() - 50000).toISOString() }
    ],
    "Another Session": []
};
let currentHistorySessionName = "Default Session";

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
    if (currentAnswerContent) {
        if (queryToShow || answerToShow) {
            const modelIcon = '../lib/llama.png';
            currentAnswerContent.innerHTML = `
                 <div class="question-container" id="answer-query-display">
                      <img src="../lib/employee.png" alt="Question Icon" class="question-icon">
                      <p class="question-text">${queryToShow || "N/A"}</p>
                 </div>
                 <div class="answer-text" id="answer-text-display">
                     <img src="${modelIcon}" alt="Model Icon" class="answer-icon-llama">
                     <span>${answerToShow || '此模式下无可用答案。'}</span>
                 </div>
            `;
        } else {
            currentAnswerContent.innerHTML = placeholderText;
        }
    } else {
        console.error("#current-answer-content 元素未找到");
    }
}

async function fetchAndDisplaySuggestions() {
    adviceContent.innerHTML = "正在加载建议...";
    try {
        // #backend-integration: 从后端 /get_suggestions 接口获取建议
        const response = await fetch('/get_suggestions');
        if (!response.ok) {
            const errorText = await response.text();
            throw new Error(`网络错误: ${response.status} ${errorText}`);
        }
        const data = await response.json();
        console.log("建议数据:", data);
        if (data.suggestionsHTML) {
            adviceContent.innerHTML = data.suggestionsHTML;
        } else if (data.advice) {
            adviceContent.innerHTML = `
                <h3>向量 RAG 错误:</h3>
                <ul>
                    <li>检索错误: ${data.vector_retrieve_error ?? 'N/A'}</li>
                    <li>丢失错误: ${data.vector_lose_error ?? 'N/A'}</li>
                    <li>丢失正确: ${data.vector_lose_correct ?? 'N/A'}</li>
                </ul>
                <h3>图谱 RAG 错误:</h3>
                <ul>
                    <li>检索错误: ${data.graph_retrieve_error ?? 'N/A'}</li>
                    <li>丢失错误: ${data.graph_lose_error ?? 'N/A'}</li>
                    <li>丢失正确: ${data.graph_lose_correct ?? 'N/A'}</li>
                </ul>
                <h3>建议:</h3>
                <p>${data.advice}</p>
            `;
        } else {
            adviceContent.textContent = "收到的建议数据格式不正确。";
            console.warn("未预期的建议数据格式", data);
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
        selectedDatasetsList.innerHTML = datasets.map(ds => `<li class="dataset-option" data-dataset-name="${ds}">${ds}</li>`).join('');
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
}

// --- 历史会话管理 ---

async function initializeHistory() {
    if (USE_BACKEND_HISTORY) {
        console.log("从后端初始化历史记录...");
        await fetchSessionsAPI();
        await populateSessionDropdown();
        await displaySessionHistory();
    } else {
        console.log("从本地模拟数据初始化历史记录...");
        populateSessionDropdown();
        displaySessionHistory();
    }
}

async function fetchSessionsAPI() {
    // #backend-integration: GET /api/sessions
    console.log("(API 模式) 正在获取会话列表...");
    try {
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

function populateSessionDropdown() {
    historySessionSelect.innerHTML = '';
    if (USE_BACKEND_HISTORY) {
        console.log("(API 模式) 正在根据 API 数据填充下拉菜单");
        if (sessionsList.length === 0) { historySessionSelect.innerHTML = '<option value="">无可用会话</option>'; return; }
        sessionsList.forEach(session => {
            const option = document.createElement('option');
            option.value = session.id; option.textContent = session.name; historySessionSelect.appendChild(option);
        });
        if (currentSessionId && sessionsList.some(s => s.id === currentSessionId)) { historySessionSelect.value = currentSessionId; }
        else if (sessionsList.length > 0) { currentSessionId = sessionsList[0].id; historySessionSelect.value = currentSessionId; }
        else { currentSessionId = null; }
        console.log("(API 模式) 下拉菜单已填充，当前会话 ID:", currentSessionId);
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
        const sessionId = currentSessionId;
        if (!sessionId) { questionList.innerHTML = '<li class="no-history-item">请选择一个会话。</li>'; return; }
        console.log(`(API 模式) 正在为会话获取历史记录: ${sessionId}`);
        // #backend-integration: GET /api/sessions/${sessionId}/history
        try {
             // const response = await fetch(`/api/sessions/${sessionId}/history`); // 实际 Fetch
             // 模拟返回
             const mockBackendHistory = [ {id: `item-be-1-${sessionId}`, query: `Backend Query 1 for ${sessionId}`, answer: "Backend Answer 1", type:"GREEN", timestamp: new Date().toISOString()} ];
             const response = { ok: true, json: async () => mockBackendHistory };
             if (!response.ok) throw new Error(`获取失败: ${response.status}`);
             items = await response.json();
             console.log(`(API 模式) 获取到 ${items.length} 条历史记录。`);
        } catch (error) {
             console.error(`(API 模式) 获取会话 ${sessionId} 的历史记录时出错:`, error);
             questionList.innerHTML = `<li class="no-history-item">加载历史记录出错: ${error.message}</li>`; return;
        }
    } else {
        const sessionName = currentHistorySessionName;
        if (!sessionName || !historySessions[sessionName]) { questionList.innerHTML = '<li class="no-history-item">请选择一个有效的会话。</li>'; return; }
        console.log(`(本地模式) 正在显示本地会话的历史记录: ${sessionName}`);
        items = (historySessions[sessionName] || []).slice().sort((a, b) => new Date(b.timestamp) - new Date(a.timestamp));
        console.log(`(本地模式) 找到 ${items.length} 条历史记录。`);
    }

    questionList.innerHTML = '';
    if (items.length === 0) { questionList.innerHTML = '<li class="no-history-item">此会话尚无历史记录。</li>'; return; }
    items.forEach(item => {
        const div = document.createElement('div'); div.classList.add('question-item'); div.id = `history-${item.id}`; div.dataset.itemId = item.id;
        let backgroundColor = '#f0f0f0'; switch (item.type?.toUpperCase()) { case 'GREEN': backgroundColor = '#d9f7be'; break; case 'RED': backgroundColor = '#ffccc7'; break; case 'YELLOW': backgroundColor = '#fff2e8'; break; } div.style.backgroundColor = backgroundColor;
        const answerSnippet = item.answer ? item.answer.substring(0, 30) + '...' : '';
        div.innerHTML = `<p>ID: ${item.id}</p><p>Query: ${item.query || 'N/A'}</p>${answerSnippet ? `<p>Ans: ${answerSnippet}</p>` : ''}`;
        div.addEventListener('click', handleHistoryItemClick); questionList.appendChild(div);
    });
}

async function handleConfirmNewSession() {
    const name = newSessionNameInput.value.trim();
    if (name === "") { console.warn("会话名称不能为空。"); newSessionNameInput.focus(); return; }
    hideNewSessionInput();
    if (USE_BACKEND_HISTORY) {
        console.log(`(API 模式) 尝试通过 API 创建新会话: ${name}`);
        // #backend-integration: POST /api/sessions
         try {
             // const response = await fetch('/api/sessions', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ name: name }) }); // 实际 Fetch
             // 模拟成功
             const newSession = { id: `backend-uuid-${Date.now()}`, name: name };
             const response = { ok: true, json: async () => newSession };
             if (!response.ok) throw new Error(`创建失败: ${response.status}`);
             const createdSession = await response.json();
             console.log("(API 模式) 新会话已创建:", createdSession);
             await fetchSessionsAPI(); currentSessionId = createdSession.id; await populateSessionDropdown(); await displaySessionHistory();
         } catch (error) { console.error("(API 模式) 创建新会话时出错:", error); alert(`通过 API 创建会话失败: ${error.message}`); }
    } else {
        console.log(`(本地模式) 尝试创建本地会话: ${name}`);
        let newName = name; let counter = 1; const baseName = newName;
        while (historySessions.hasOwnProperty(newName)) { newName = `${baseName} ${counter}`; counter++; }
        console.log(`(本地模式) 创建本地会话: ${newName}`);
        historySessions[newName] = []; currentHistorySessionName = newName;
        populateSessionDropdown(); displaySessionHistory();
    }
}

async function addInteractionToHistory(query, answer, type = 'INFO', details = {}) {
    const historyItemData = {
        query: query, answer: answer, type: type,
        details: { vectorAnswer: details.vectorAnswer || '', graphAnswer: details.graphAnswer || '', hybridAnswer: details.hybridAnswer || '' }
    };
    if (USE_BACKEND_HISTORY) {
        const sessionId = currentSessionId;
        if (!sessionId) { console.error("(API 模式) 无法添加到历史: 未选择会话。"); return null; }
        console.log(`(API 模式) 正在添加交互到会话 ${sessionId}`);
        // #backend-integration: POST /api/sessions/${sessionId}/history
         try {
             // const response = await fetch(`/api/sessions/${sessionId}/history`, { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(historyItemData) }); // 实际 Fetch
             // 模拟成功
             const savedItem = { ...historyItemData, id: `item-be-${Date.now()}-${sessionId}`, timestamp: new Date().toISOString() };
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

function showNewSessionInput() {
    if (isAddingNewSession) return;
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

    
    applySettingsButton.disabled = true; 
    applySettingsButton.textContent = "应用中..."; 
    adviceContent.innerHTML = "正在加载建议...";

    try {
        const postData = {
            hop: hop,
            type: type,
            entity: entity,
            dataset: selectedDatasetName
        };
        fetch('/get-dataset', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify(postData)
        }).catch(error => {
            console.error("请求失败:", error);
        });
    
        console.log("数据集加载请求已发送（POST），不等待结果。");
    
    } catch (error) {
        console.error("构造请求失败:", error);
        alert("发生错误，无法发送数据集请求。");
    } finally {
        applySettingsButton.disabled = false; 
        applySettingsButton.textContent = "应用设置"; 
    }

    
    const settingsData = { 
        model_name: modelSelect.value, 
        dataset: selectedDatasetName, 
        key: apiKeyInput.value, 
        top_k: parseInt(document.getElementById("top-k").value) || 5, 
        threshold: parseFloat(document.getElementById("similarity-threshold").value) || 0.8, 
        chunksize: parseInt(document.getElementById("chunk-size").value) || 128, 
        k_hop: parseInt(document.getElementById("k-hop").value) || 1, 
        max_keywords: parseInt(document.getElementById("max-keywords").value) || 10, 
        pruning: document.getElementById("pruning").value === "yes", 
        strategy: document.getElementById("strategy").value || "union", 
        vector_proportion: parseFloat(document.getElementById("vector-proportion").value) || 0.9, 
        graph_proportion: parseFloat(document.getElementById("graph-proportion").value) || 0.8 
    };
    
    console.log("正在应用设置:", settingsData);
    
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
        // await fetchAndDisplaySuggestions();
        
    } catch (error) { 
        console.error("应用设置时发生错误:", error); 
        alert(`发生错误: ${error.message}`); 
        adviceContent.innerHTML = `应用设置时发生错误: ${error.message}`; 
    } finally { 
        applySettingsButton.disabled = false; 
        applySettingsButton.textContent = "Apply Settings"; 
    }
});

// --- 检索结果显示逻辑 ---

async function handleHistoryItemClick(event) {
     const div = event.currentTarget; const itemId = div.dataset.itemId; if (!itemId) return;
     let clickedItemData = null; let queryText = 'Loading...';

     if (USE_BACKEND_HISTORY) {
         queryText = div.querySelector('p:nth-of-type(2)')?.textContent.replace('Query: ', '') || 'Loading...';
         // #backend-integration: Optionally fetch full item details from backend if needed
         // For now, assume we only have ID and query from the list item
         clickedItemData = { id: itemId, query: queryText }; // Minimal data
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
     updateAnswerStore(clickedItemData); displaySelectedAnswer();

     // #backend-integration: GET /get-vector/${itemId} 和 GET /get-graph/${itemId}
     // Backend needs to handle UUIDs if USE_BACKEND_HISTORY=true, or mock/local IDs if false
     let vectorResponse, graphResponse;
     try { console.log(`正在为项 ID 获取细节: ${itemId}`); [vectorResponse, graphResponse] = await Promise.all([ fetch(`/get-vector/${itemId}`), fetch(`/get-graph/${itemId}`) ]);
         if (vectorResponse.ok) { const d = await vectorResponse.json(); if (d?.chunks) { vectorContent.innerHTML = d.chunks.map((c, i) => `<div class="retrieval-result-item"><p><b>Chunk ${i + 1}:</b> ${c.text || 'N/A'}</p></div>`).join(''); } else { vectorContent.innerHTML = '<p>未找到向量块。</p>'; } } else { const t = await vectorResponse.text(); vectorContent.innerHTML = `<p>向量错误 ${vectorResponse.status}: ${t}</p>`; console.error(`向量 Fetch 失败: ${t}`);}
         if (graphResponse.ok) { const d = await graphResponse.json(); if (d?.nodes || d?.edges) { renderCytoscapeGraph(d); } else { cyGraphDiv.innerHTML = '<p>未找到图谱数据。</p>'; } } else { const t = await graphResponse.text(); cyGraphDiv.innerHTML = `<p>图谱错误 ${graphResponse.status}: ${t}</p>`; console.error(`图谱 Fetch 失败: ${t}`);}
     } catch (error) { console.error(`为项 ${itemId} 获取细节时出错:`, error); if (!vectorResponse?.ok) vectorContent.innerHTML = `<p>加载向量细节失败。 ${error.message}</p>`; if (!graphResponse?.ok) cyGraphDiv.innerHTML = `<p>加载图谱细节失败。 ${error.message}</p>`; }
}

function renderCytoscapeGraph(graphData) {
    let cyTargetDiv = document.getElementById('cy'); if (!cyTargetDiv) { console.error("Cytoscape 容器 'cy' 在 DOM 中未找到。"); cyContainer.innerHTML = ''; cyTargetDiv = document.createElement('div'); cyTargetDiv.id = 'cy'; cyContainer.appendChild(cyTargetDiv); } else { cyTargetDiv.innerHTML = ''; } if (currentCytoscapeInstance) { currentCytoscapeInstance.destroy(); currentCytoscapeInstance = null; } try { currentCytoscapeInstance = cytoscape({ container: cyTargetDiv, elements: { nodes: graphData.nodes || [], edges: graphData.edges || [] }, style: [ { selector: 'node', style: { 'background-color': 'data(color, "#888")', 'label': 'data(label, id)', 'width': 50, 'height': 50, 'font-size': '10px', 'text-valign': 'center', 'text-halign': 'center', 'color': '#000', 'text-outline-color': '#fff', 'text-outline-width': 1 } }, { selector: 'edge', style: { 'line-color': 'data(color, "#ccc")', 'target-arrow-color': 'data(color, "#ccc")', 'curve-style': 'bezier', 'target-arrow-shape': 'triangle', 'label': 'data(label)', 'width': 2 , 'font-size': '8px', 'text-rotation': 'autorotate', 'text-margin-y': -5, 'color': '#000', 'text-background-color': '#fff', 'text-background-opacity': 0.7, 'text-background-padding': '1px'} }, { selector: '.highlighted-node', style: { 'background-color': '#FF5733', 'border-color': '#E84A27', 'border-width': 3, 'width': 60, 'height': 60, 'z-index': 10, 'shadow-blur': 10, 'shadow-color': '#FF5733', 'shadow-opacity': 0.8 } }, { selector: '.highlighted-edge', style: { 'line-color': '#FF5733', 'target-arrow-color': '#FF5733', 'width': 4, 'z-index': 9, 'shadow-blur': 5, 'shadow-color': '#FF5733', 'shadow-opacity': 0.6 } } ], layout: { name: 'cose', fit: true, padding: 30, animate: true, animationDuration: 500, nodeRepulsion: 400000, idealEdgeLength: 100, nodeOverlap: 20 } }); if (graphData['highlighted-node']?.forEach) { graphData['highlighted-node'].forEach(n => n?.data?.id && currentCytoscapeInstance.getElementById(n.data.id).addClass('highlighted-node')); } if (graphData['highlighted-edge']?.forEach) { graphData['highlighted-edge'].forEach(e => e?.data?.id && currentCytoscapeInstance.getElementById(e.data.id).addClass('highlighted-edge')); } currentCytoscapeInstance.ready(() => { currentCytoscapeInstance.fit(null, 30); }); console.log("Cytoscape 图谱已渲染。"); } catch (error) { console.error("Cytoscape 渲染错误:", error); cyTargetDiv.innerHTML = `<p>渲染图谱时出错: ${error.message}</p>`; currentCytoscapeInstance = null; }
}

function updateAnswerStore(data) {
    currentAnswers.query = data.query !== undefined ? data.query : currentAnswers.query;
    currentAnswers.vector = data.vectorAnswer !== undefined ? data.vectorAnswer : "";
    currentAnswers.graph = data.graphAnswer !== undefined ? data.graphAnswer : "";
    currentAnswers.hybrid = data.hybridAnswer !== undefined ? data.hybridAnswer : "";
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

// --- 初始化 ---
document.addEventListener('DOMContentLoaded', async () => {
    console.log("DOM 完全加载并解析。");
    document.querySelectorAll('.sidebar-section .sidebar-header, .sidebar-section-inner .sidebar-header-inner').forEach((header) => { 
        const content = header.nextElementSibling; const icon = header.querySelector('.material-icons');
        if (!header.classList.contains('collapsed')) header.classList.add("collapsed");
        if (content) content.style.display = "none";
        if (icon) icon.textContent = 'expand_more'; 
    }
);
    populateSelect(dim1Select, Object.keys(datasetHierarchy));
    clearSelect(dim2Select);
    clearSelect(dim3Select);
    updateDatasetSelection();
    applySettingsButton.disabled = true;
    await initializeHistory(); // 初始化历史记录区域 (根据 USE_BACKEND_HISTORY 决定行为)
    displaySelectedAnswer();
});

// --- 事件监听器 ---
ragSelect.addEventListener("change", displaySelectedAnswer);
dim1Select.addEventListener('change', updateDatasetSelection);
dim2Select.addEventListener('change', updateDatasetSelection);
dim3Select.addEventListener('change', updateDatasetSelection);
historySessionSelect.addEventListener('change', (event) => {
    const selectedValue = event.target.value;
    if (USE_BACKEND_HISTORY) { currentSessionId = selectedValue; console.log("(API 模式) 会话已更改为 ID:", currentSessionId); }
    else { currentHistorySessionName = selectedValue; console.log("(本地模式) 会话已更改为名称:", currentHistorySessionName); }
    displaySessionHistory();
});
newHistorySessionButton.addEventListener('click', showNewSessionInput);
newSessionNameInput.addEventListener('keydown', (event) => { if (event.key === 'Enter') { event.preventDefault(); handleConfirmNewSession(); } else if (event.key === 'Escape') { hideNewSessionInput(); } });
cancelNewSessionButton.addEventListener('click', hideNewSessionInput);