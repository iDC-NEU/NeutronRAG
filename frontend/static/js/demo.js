// --- 全局配置 & 状态 ---
let isGenerating = false;
let abortController = new AbortController();
let currentCytoscapeInstance = null;
let currentAnswers = { query: "", vector: "", graph: "", hybrid: "" };
let selectedDatasetName = null; // Dataset name selected in settings
let sessionsList = []; // List of {id, name} for the current user's sessions
let currentSessionId = null; // ID of the currently selected chat session
let currentUser = { logged_in: false, username: null }; // Store user auth state
let currentMessageId = null; // Track the message ID being displayed (from history click)

// --- 常量与 DOM 元素 ---
const sendButton = document.getElementById("send-button");
const applySettingsButton = document.getElementById("applySettingsButton");
const userInput = document.getElementById("user-input");
const ragSelect = document.getElementById("rag-select");
const currentAnswerContent = document.getElementById('current-answer-content');
const adviceContent = document.getElementById("advice-text");
const vectorContent = document.getElementById("vector-content");
const cyContainer = document.getElementById('cy-container'); // Changed selector to container
const questionList = document.getElementById("question-list"); // Displays messages of current session
const modelSelect = document.getElementById("model-select");
const apiKeyInput = document.getElementById("api-key-input");
const dim1Select = document.getElementById('dim1-hops');
const dim2Select = document.getElementById('dim2-task');
const dim3Select = document.getElementById('dim3-scale');
const selectedDatasetsList = document.getElementById('selected-datasets-list');
const historySessionSelect = document.getElementById('history-session-select'); // Dropdown to select session
const newHistorySessionButton = document.getElementById('new-history-session-button');
const newSessionInputContainer = document.getElementById('new-session-input-container');
const newSessionNameInput = document.getElementById('new-session-name-input');
const cancelNewSessionButton = document.getElementById('cancel-new-session-button');

// Placeholders for user info/logout from demo.html
const userInfoDisplay = document.getElementById('user-info-display');
const logoutButton = document.getElementById('logout-button');

const placeholderText = `<div class="placeholder-text">请选择 RAG 模式，输入内容或从历史记录中选择。</div>`;

// --- 数据集层级结构 (Keep as is) ---
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
}; // Make sure this reflects your actual data structure

// --- UI 辅助函数 (Keep as is or adapt) ---
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

// --- 认证与初始化 ---

async function checkAuth() {
    // Checks login status when the page loads
    try {
        const response = await fetch('/api/check-auth'); // API endpoint to check session
        if (!response.ok) {
            // Assume not logged in if check fails for any reason other than explicit 'logged_in: false'
            console.warn(`Auth check failed: ${response.status}, assuming logged out.`);
            currentUser = { logged_in: false };
        } else {
             currentUser = await response.json();
        }


        if (!currentUser.logged_in) {
            console.log("User not logged in. Redirecting to login.");
            window.location.href = '/login'; // Redirect if not logged in
        } else {
            console.log(`User logged in: ${currentUser.username}`);
            updateUserInfoUI(); // Show username and logout button
            await initializeDemo(); // Load sessions etc. only if logged in
        }
    } catch (error) {
        console.error("Error checking authentication:", error);
        // Redirect to login on any error during auth check
        alert("无法验证登录状态，请重新登录。");
        window.location.href = '/login';
    }
}

function updateUserInfoUI() {
    // Updates the header UI based on login status
    if (currentUser.logged_in && userInfoDisplay) {
        userInfoDisplay.textContent = `欢迎, ${currentUser.username}`;
        userInfoDisplay.style.display = 'inline'; // Show username
    } else if (userInfoDisplay) {
         userInfoDisplay.style.display = 'none'; // Hide username
    }
     if (logoutButton) {
         logoutButton.style.display = currentUser.logged_in ? 'inline-block' : 'none'; // Show/hide logout button
     }
}

async function handleLogout() {
    // Called when logout button is clicked
    console.log("Logging out...");
    try {
        const response = await fetch('/api/logout', { method: 'POST' });
        if (!response.ok) {
             // Try to get error message from response body
             const errData = await response.json().catch(() => ({})); // Default empty if JSON parse fails
             throw new Error(errData.error || `Logout failed: ${response.status}`);
        }
        const data = await response.json();
        console.log(data.message); // "注销成功"
        currentUser = { logged_in: false, username: null }; // Update local state
        window.location.href = '/login'; // Redirect to login page
    } catch (error) {
        console.error("Logout error:", error);
        alert(`登出时出错: ${error.message}`);
    }
}

async function initializeDemo() {
    // Initial setup after successful login check
    console.log("Initializing demo for logged in user...");
    // Initial UI setup (collapsing sections etc.)
    document.querySelectorAll('.sidebar-section .sidebar-header, .sidebar-section-inner .sidebar-header-inner').forEach((header) => {
        const content = header.nextElementSibling; const icon = header.querySelector('.material-icons');
        if (!header.classList.contains('collapsed')) header.classList.add("collapsed");
        if (content) content.style.display = "none";
        if (icon) icon.textContent = 'expand_more';
    });
    populateSelect(dim1Select, Object.keys(datasetHierarchy));
    clearSelect(dim2Select);
    clearSelect(dim3Select);
    updateDatasetSelection(); // Handles dependent dropdowns
    applySettingsButton.disabled = true; // Disable until dataset is selected

    await fetchUserSessions(); // Load user's sessions into dropdown
    displaySelectedAnswer(); // Show initial placeholder in answer area
    clearRetrievalResults(); // Clear retrieval areas initially
}

// --- 核心 RAG & 交互逻辑 ---

function displaySelectedAnswer() {
    // Displays the query and the answer for the selected RAG mode
    const selectedMode = ragSelect.value;
    // Map backend fields to frontend modes if necessary (assuming direct match here)
    const answerMap = {
        'vector': currentAnswers.vector,
        'graph': currentAnswers.graph,
        'hybrid': currentAnswers.hybrid
    };
    const answerToShow = answerMap[selectedMode];
    const queryToShow = currentAnswers.query;
    const answerContentElement = document.getElementById('current-answer-content');

    if (answerContentElement) {
        let chatHTML = '';
        // Display user query bubble
        if (queryToShow) {
             // Use url_for in JS is tricky, use relative or absolute paths known at build/deploy time
             // Assuming static files are served from /static/
             chatHTML += `
                <div class="chat-message user-message">
                    <img src="/static/lib/employee.png" alt="User Icon" class="message-icon user-icon-bubble">
                    <div class="message-bubble">${queryToShow}</div>
                </div>`;
        }
        // Display model answer bubble
        if (answerToShow) {
             const modelIcon = '/static/lib/llama.png'; // Adjust path if needed
             chatHTML += `
                <div class="chat-message model-message">
                    <img src="${modelIcon}" alt="Model Icon" class="message-icon model-icon-bubble">
                    <div class="message-bubble">${answerToShow}</div>
                </div>`;
        }
        // Display placeholder if no query/answer yet
        if (!queryToShow && !answerToShow) {
             chatHTML = `<div class="placeholder-text">请输入问题并选择模式以开始，或从历史记录中选择。</div>`;
        }
        answerContentElement.innerHTML = chatHTML;
        // Scroll to the bottom of the chat window
        answerContentElement.scrollTop = answerContentElement.scrollHeight;
    } else {
        console.error("#current-answer-content 元素未找到");
    }
}

sendButton.addEventListener("click", async () => {
    // Handles sending user input to the backend for generation
    if (isGenerating) {
        // If already generating, the button acts as a cancel button
        abortController.abort();
        console.log("Generation aborted by user.");
        // Reset button state in finally block
        return;
    }
    if (!currentSessionId) {
        alert("请先选择或创建一个聊天会话。");
        return;
    }
    const query = userInput.value.trim();
    if (!query) {
        alert("请输入查询内容。");
        return;
    }

    isGenerating = true;
    sendButton.textContent = "取消"; // Change button text
    sendButton.classList.add("generating"); // Optional: for styling
    userInput.disabled = true; // Disable input during generation
    abortController = new AbortController(); // Create a new controller for this request

    // Clear previous results, show query and loading state
    currentAnswers = { query: query, vector: "生成中...", graph: "生成中...", hybrid: "生成中..." };
    displaySelectedAnswer();
    clearRetrievalResults(true); // Show loading indicators in retrieval areas

    let generatedData = {};
    let fetchError = null;

    try {
        // Call the backend /generate endpoint
        const response = await fetch("/generate", {
            method: "POST",
            headers: { "Content-Type": "application/json", "Accept": "application/json" },
            body: JSON.stringify({
                input: query,
                session_id: currentSessionId,
                rag_mode: ragSelect.value // Send selected RAG mode
            }),
            signal: abortController.signal // Pass the abort signal
        });

        if (response.ok) {
            generatedData = await response.json();
            console.log("Generated data received:", generatedData);
            // Update internal state with received answers
            updateAnswerStore(generatedData);
            // Display the actual answers
            displaySelectedAnswer();
            // Refresh the history list automatically to show the new message
            await displaySessionHistory();
            // Scroll to and highlight the new item in history (optional)
            const newItemElement = document.getElementById(`message-${generatedData.message_id}`);
            if (newItemElement) {
                newItemElement.click(); // Simulate click to load details
                newItemElement.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
            }

        } else {
            // Handle HTTP errors (e.g., 400, 500)
            const errorText = await response.text();
            fetchError = new Error(`生成失败: ${response.status} ${errorText}`);
            currentAnswers = { query: query, vector: "错误", graph: "错误", hybrid: "错误" }; // Show error state
            displaySelectedAnswer();
        }
    } catch (error) {
        fetchError = error;
        if (error.name === "AbortError") {
            // Handle cancellation
            console.log("Fetch aborted by user.");
            currentAnswers = { query: query, vector: "已取消", graph: "已取消", hybrid: "已取消" };
        } else {
            // Handle other network or unexpected errors
            console.error("Fetch /generate error:", error);
            currentAnswers = { query: query, vector: "错误", graph: "错误", hybrid: "错误" };
             // Update answer display with more specific error if possible
             if (!error.message?.includes('aborted')) {
                 currentAnswers = { query: query, vector: `错误: ${error.message}`, graph: `错误: ${error.message}`, hybrid: `错误: ${error.message}` };
             }
        }
         displaySelectedAnswer(); // Display error/cancel state
    } finally {
        // Reset state regardless of success, failure, or cancellation
        isGenerating = false;
        sendButton.textContent = "Send";
        sendButton.classList.remove("generating");
        userInput.disabled = false;
        // Log or alert if there was an error (and it wasn't an abort)
        if (fetchError && fetchError.name !== "AbortError") {
            console.error("Generation process failed:", fetchError);
            alert(`生成过程中遇到问题: ${fetchError.message}`);
        }
    }
});

// --- Settings Application ---
applySettingsButton.addEventListener("click", async () => {
    // Handles applying settings from the sidebar
    if (!selectedDatasetName) {
        alert("请在选择所有维度后，从列表中选择一个数据集。");
        return;
    }
    const hop = dim1Select.value;
    const type = dim2Select.value;
    const entity = dim3Select.value;
    if (!hop || !type || !entity) {
        alert("请完整选择数据集维度 (Hops, Task, Scale)！");
        return;
    }

    applySettingsButton.disabled = true;
    applySettingsButton.textContent = "应用中...";
    adviceContent.innerHTML = "正在应用设置并加载模型..."; // Update status message

    // Collect all settings data
    const settingsData = {
        dataset: { // Send dataset info
            hop: hop,
            type: type,
            entity: entity,
            dataset: selectedDatasetName,
        },
        model_name: modelSelect.value,
        key: apiKeyInput.value,
        // RAG parameters
        top_k: parseInt(document.getElementById("top-k").value) || 5,
        threshold: parseFloat(document.getElementById("similarity-threshold").value) || 0.8,
        chunksize: parseInt(document.getElementById("chunk-size").value) || 128,
        k_hop: parseInt(document.getElementById("k-hop").value) || 1,
        max_keywords: parseInt(document.getElementById("max-keywords").value) || 10,
        pruning: document.getElementById("pruning").value === "yes", // Convert to boolean
        strategy: document.getElementById("strategy").value || "union",
        vector_proportion: parseFloat(document.getElementById("vector-proportion").value) || 0.9,
        graph_proportion: parseFloat(document.getElementById("graph-proportion").value) || 0.8
    };

    console.log("Applying settings:", settingsData);

    try {
        // Call the backend /load_model endpoint
        const response = await fetch("/load_model", {
            method: "POST",
            headers: { "Content-Type": "application/json", "Accept": "application/json" },
            body: JSON.stringify(settingsData)
        });

        const result = await response.json(); // Try parsing JSON response

        if (!response.ok) {
            // Throw error with message from backend if available
            throw new Error(result.message || `应用设置失败: ${response.status}`);
        }

        console.log("Settings applied successfully:", result);
        alert(result.message || "设置应用成功！"); // Show success feedback
        adviceContent.innerHTML = "设置已应用。"; // Update status

        // Optional: Fetch suggestions after applying settings
        // await fetchAndDisplaySuggestions();

    } catch (error) {
        console.error("应用设置时发生错误:", error);
        alert(`应用设置时出错: ${error.message}`); // Show error feedback
        adviceContent.innerHTML = `应用设置时发生错误: ${error.message}`; // Update status
    } finally {
        applySettingsButton.disabled = false; // Re-enable button
        applySettingsButton.textContent = "Apply Settings";
    }
});


// --- Dataset Dimension Selection ---
function updateDatasetSelection() {
    // Updates the dataset selection dropdowns based on hierarchy
    const dim1Value = dim1Select.value;
    const dim2Value = dim2Select.value;
    const dim3Value = dim3Select.value;
    let datasets = [];
    selectedDatasetName = null; // Reset selected dataset name
    applySettingsButton.disabled = true; // Disable apply button until a dataset is clicked
    selectedDatasetsList.innerHTML = '<li>请先选择以上维度...</li>'; // Reset dataset list

    // --- Logic to populate dim2, dim3 based on dim1, dim2 values ---
    // (This part remains the same as your original logic)
    if (!dim1Value) { clearSelect(dim2Select); clearSelect(dim3Select); return; }
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
        datasets = level3; // datasets is the array of dataset names
    } catch (e) { console.error("导航数据集层级时出错:", e); datasets = []; selectedDatasetsList.innerHTML = '<li>选择出错。</li>'; return; }
    // --- End population logic ---

    // Display the list of selectable datasets
    if (Array.isArray(datasets) && datasets.length > 0) {
        selectedDatasetsList.innerHTML = datasets.map(ds => `<li class="dataset-option" data-dataset-name="${ds}">${ds}</li>`).join('');
        // Add click listener to each dataset list item
        selectedDatasetsList.querySelectorAll('.dataset-option').forEach(item => item.addEventListener('click', handleDatasetOptionClick));
        selectedDatasetsList.insertAdjacentHTML('afterbegin', '<li>请点击选择一个数据集:</li>');
    } else {
        selectedDatasetsList.innerHTML = '<li>此维度组合下未找到数据集。</li>';
    }
}

function handleDatasetOptionClick(event) {
    // Handles clicking on a dataset name in the list
    const li = event.currentTarget;
    const datasetName = li.dataset.datasetName;

    // Remove highlight from previously selected item
    const currentlySelected = selectedDatasetsList.querySelector('.selected-dataset');
    if (currentlySelected) { currentlySelected.classList.remove('selected-dataset'); }

    // Highlight the newly selected item
    li.classList.add('selected-dataset');
    selectedDatasetName = datasetName; // Store the selected name
    console.log("选择的数据集:", selectedDatasetName);

    // Enable the 'Apply Settings' button now that a dataset is chosen
    applySettingsButton.disabled = false;
}

// --- Helper functions for dataset dropdowns (Keep as is) ---
function populateSelect(selectElement, options) {
    const currentVal = selectElement.value;
    const defaultOptionText = selectElement.options[0]?.textContent || `-- 请选择 --`; // Preserve default text
    selectElement.innerHTML = `<option value="">${defaultOptionText}</option>`; // Reset with default
    options.forEach(option => {
        const opt = document.createElement('option');
        opt.value = option;
        opt.textContent = option;
        selectElement.appendChild(opt);
    });
    // Restore previous selection if still valid
    if (options.includes(currentVal)) {
        selectElement.value = currentVal;
    } else {
        selectElement.value = ""; // Reset if previous value is no longer valid
    }
}
function clearSelect(selectElement, keepDisabled = true) {
    const defaultOptionText = selectElement.options[0]?.textContent || `-- 请选择 --`;
    selectElement.innerHTML = `<option value="">${defaultOptionText}</option>`;
    selectElement.disabled = keepDisabled;
}

// --- History / Session Management ---

async function fetchUserSessions() {
    // Fetches the list of chat sessions for the logged-in user
    console.log("Fetching user sessions...");
    try {
        const response = await fetch('/api/sessions'); // GET request to list sessions
        if (!response.ok) throw new Error(`Failed to fetch sessions: ${response.status}`);
        sessionsList = await response.json(); // Expecting [{id, name, create_time}, ...]
        console.log("Sessions loaded:", sessionsList);
        populateSessionDropdown();

        // Automatically select the first session if available
        if (sessionsList.length > 0) {
             // Default to the first session in the list (which should be the most recent if backend sorts)
             currentSessionId = sessionsList[0].id;
             historySessionSelect.value = currentSessionId; // Update dropdown display
             await displaySessionHistory(); // Load history for the selected session
        } else {
            // Handle case where user has no sessions yet
            currentSessionId = null;
            historySessionSelect.innerHTML = '<option value="">无会话</option>';
            questionList.innerHTML = '<li class="no-history-item">无可用会话。请创建一个新会话开始。</li>';
        }
    } catch (error) {
        console.error("Error fetching sessions:", error);
        historySessionSelect.innerHTML = '<option value="">加载出错</option>';
        questionList.innerHTML = `<li class="no-history-item">加载会话列表出错: ${error.message}</li>`;
    }
}

function populateSessionDropdown() {
    // Fills the session dropdown (#history-session-select)
    historySessionSelect.innerHTML = ''; // Clear existing options
    if (sessionsList.length === 0) {
        historySessionSelect.innerHTML = '<option value="">无会话</option>';
        return;
    }
    // Sort sessions by create_time descending if backend didn't already
    // sessionsList.sort((a, b) => new Date(b.create_time) - new Date(a.create_time));

    sessionsList.forEach(session => {
        const option = document.createElement('option');
        option.value = session.id; // Use session ID as the value
        option.textContent = session.name; // Display session name
        historySessionSelect.appendChild(option);
    });

    // Ensure the dropdown reflects the currently selected session ID
    if (currentSessionId && sessionsList.some(s => s.id === currentSessionId)) {
        historySessionSelect.value = currentSessionId;
    } else if (sessionsList.length > 0) {
         // If no valid currentSessionId, default to the first in the list
         currentSessionId = sessionsList[0].id;
         historySessionSelect.value = currentSessionId;
    } else {
        currentSessionId = null; // No sessions available
    }
}

historySessionSelect.addEventListener('change', async (event) => {
    // Handles changing the selected session in the dropdown
    const selectedId = parseInt(event.target.value); // Ensure ID is number if needed
    if (selectedId && selectedId !== currentSessionId) {
        currentSessionId = selectedId;
        console.log("Session changed to ID:", currentSessionId);
        await displaySessionHistory(); // Load history for the newly selected session

        // Clear main answer/retrieval areas when switching sessions
        currentAnswers = { query: "", vector: "", graph: "", hybrid: "" };
        displaySelectedAnswer();
        clearRetrievalResults();
        currentMessageId = null; // Reset selected message ID
    }
});

async function displaySessionHistory() {
    // Fetches and displays messages for the currentSessionId
    if (!currentSessionId) {
        questionList.innerHTML = '<li class="no-history-item">请先选择一个会话。</li>';
        return;
    }
    questionList.innerHTML = '<li class="no-history-item">正在加载历史记录...</li>';
    console.log(`Workspaceing history for session ID: ${currentSessionId}`);

    try {
        // Call backend API to get messages for the selected session
        const response = await fetch(`/api/sessions/${currentSessionId}/history`);
        if (!response.ok) {
            const errData = await response.json().catch(() => ({}));
            throw new Error(errData.error || `获取历史记录失败: ${response.status}`);
        }
        const messages = await response.json(); // Expecting array of message objects from backend
        console.log(`Loaded ${messages.length} messages for session ${currentSessionId}`);

        questionList.innerHTML = ''; // Clear loading/previous messages

        if (messages.length === 0) {
            questionList.innerHTML = '<li class="no-history-item">此会话尚无历史记录。</li>';
            return;
        }

        // Render each message item (consider ordering - backend likely sends oldest first)
        messages.forEach(item => { // Display oldest first, newest at bottom
            const div = document.createElement('div');
            div.classList.add('question-item'); // Use existing CSS class
            div.id = `message-${item.id}`; // Unique ID for the message item element
            div.dataset.messageId = item.id; // Store message ID for easy access on click

            // Display query and a short answer snippet
            const answerSnippet = (item.hybrid_response || item.vector_response || item.graph_response || "...")?.substring(0, 30) + '...';
            // Format timestamp nicely
             const timestamp = new Date(item.timestamp).toLocaleString(); // Use local time format

            // Store full answers in hidden spans within the element for quick display on click
            div.innerHTML = `
                <p class="history-query">问: ${item.query || 'N/A'}</p>
                <p class="history-answer">答: ${answerSnippet}</p>
                <p class="history-timestamp">${timestamp}</p>
                <span hidden class="data-vector">${item.vector_response || ''}</span>
                <span hidden class="data-graph">${item.graph_response || ''}</span>
                <span hidden class="data-hybrid">${item.hybrid_response || ''}</span>
            `;

            // Optional: Add styling based on message type/status if that data exists
            // switch (item.type?.toUpperCase()) { /* ... add background color logic ... */ }

            div.addEventListener('click', handleHistoryItemClick); // Add click handler
            questionList.appendChild(div);
        });
         // Scroll history list to the bottom (most recent message)
         questionList.scrollTop = questionList.scrollHeight;

    } catch (error) {
        console.error(`获取会话 ${currentSessionId} 历史记录时出错:`, error);
        questionList.innerHTML = `<li class="no-history-item">加载历史记录出错: ${error.message}</li>`;
    }
}

// --- New Session Creation ---
newHistorySessionButton.addEventListener('click', showNewSessionInput);
cancelNewSessionButton.addEventListener('click', hideNewSessionInput);

newSessionNameInput.addEventListener('keydown', (event) => {
    // Allow creating session on Enter key
    if (event.key === 'Enter') {
        event.preventDefault(); // Prevent form submission if it were inside a form
        handleConfirmNewSession();
    } else if (event.key === 'Escape') {
        // Allow canceling with Escape key
        hideNewSessionInput();
    }
});

function showNewSessionInput() {
    // Shows the input field for creating a new session name
    newHistorySessionButton.style.display = 'none'; // Hide the '+' button
    newSessionInputContainer.style.display = 'inline-flex'; // Show the input container
    newSessionNameInput.value = ''; // Clear previous input
    newSessionNameInput.focus(); // Focus the input field
}
function hideNewSessionInput() {
    // Hides the input field and shows the '+' button again
    newSessionInputContainer.style.display = 'none';
    newHistorySessionButton.style.display = 'inline-block';
}

async function handleConfirmNewSession() {
    // Creates a new session via backend API
    const name = newSessionNameInput.value.trim();
    if (name === "") {
        alert("请输入会话名称。");
        newSessionNameInput.focus();
        return;
    }
    hideNewSessionInput(); // Hide input immediately
    console.log(`Attempting to create new session: ${name}`);

    try {
        // Call the backend API to create the session
        const response = await fetch('/api/sessions', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json', "Accept": "application/json" },
            body: JSON.stringify({ sessionName: name }) // Backend expects 'sessionName'
        });
        const newSession = await response.json(); // Get the created session details {id, name, ...}

        if (!response.ok) {
            // Handle creation failure
            throw new Error(newSession.error || `创建会话失败: ${response.status}`);
        }

        console.log("New session created:", newSession);
        // Refresh the session list in the dropdown
        await fetchUserSessions();
        // Automatically select the newly created session
        currentSessionId = newSession.id;
        historySessionSelect.value = currentSessionId;
        // Display the (empty) history for the new session
        await displaySessionHistory();
        // Clear main answer area for the new session
        currentAnswers = { query: "", vector: "", graph: "", hybrid: "" };
        displaySelectedAnswer();
        clearRetrievalResults();

    } catch (error) {
        console.error("创建新会话时出错:", error);
        alert(`创建会话失败: ${error.message}`);
        // Consider showing the input again if creation fails?
        // showNewSessionInput();
    }
}

// --- Retrieval Result Display ---

async function handleHistoryItemClick(event) {
    // Handles clicking on a message item in the history list
    const div = event.currentTarget;
    const messageId = div.dataset.messageId; // Get message ID from the clicked element
    if (!messageId) {
        console.warn("Clicked history item missing message ID.");
        return;
    }

    currentMessageId = messageId; // Keep track of the currently selected message

    // Highlight the selected item in the history list
    document.querySelectorAll('.question-item.selected').forEach(el => el.classList.remove('selected'));
    div.classList.add('selected');

    console.log(`History item clicked, Message ID: ${messageId}`);

    // --- Update Main Answer Display ---
    // Retrieve the full query and answers stored within the clicked element
    const queryText = div.querySelector('.history-query')?.textContent.replace('问: ', '') || 'N/A';
    currentAnswers.query = queryText;
    currentAnswers.vector = div.querySelector('.data-vector')?.textContent || '';
    currentAnswers.graph = div.querySelector('.data-graph')?.textContent || '';
    currentAnswers.hybrid = div.querySelector('.data-hybrid')?.textContent || '';
    displaySelectedAnswer(); // Update the main chat display area
    // --- End Answer Display Update ---


    // --- Fetch and Display Detailed Retrieval Results (Vector/Graph) ---
    clearRetrievalResults(true); // Show loading state in retrieval areas

    try {
        // Fetch vector and graph details concurrently using the message ID
        const [vectorResponse, graphResponse] = await Promise.all([
            fetch(`/get-vector/${messageId}`), // API expects message ID
            fetch(`/get-graph/${messageId}`)  // API expects message ID
        ]);

        // Process Vector Results
        if (vectorResponse.ok) {
            const vectorData = await vectorResponse.json(); // Expecting { id: ..., chunks: [...] }
             // Check if chunks exist and is an array
            if (vectorData?.chunks && Array.isArray(vectorData.chunks)) {
                if (vectorData.chunks.length > 0) {
                    vectorContent.innerHTML = vectorData.chunks.map((chunkData, i) => {
                        // Adapt based on the actual structure of items in the chunks array
                        const chunkText = (typeof chunkData === 'string') ? chunkData : (chunkData.chunk || JSON.stringify(chunkData)); // Example handling
                        return `<div class="retrieval-result-item">
                                    <p><b>Chunk ${i + 1}:</b> ${chunkText || 'N/A'}</p>
                                </div>`;
                    }).join('');
                } else {
                     vectorContent.innerHTML = '<p>未找到相关向量块。</p>';
                }
            } else {
                 console.warn("Vector data received, but 'chunks' array is missing or not an array:", vectorData);
                 vectorContent.innerHTML = '<p>向量数据格式不正确。</p>';
            }
        } else {
            // Handle vector fetch error
            const errorText = await vectorResponse.text();
            vectorContent.innerHTML = `<p>加载向量数据出错 ${vectorResponse.status}: ${errorText}</p>`;
            console.error(`Vector fetch failed (${messageId}): ${errorText}`);
        }

        // Process Graph Results
         if (graphResponse.ok) {
            const graphData = await graphResponse.json(); // Expecting Cytoscape JSON { nodes: [...], edges: [...] }
             // Check if graphData is valid before rendering
            if (graphData && (Array.isArray(graphData.nodes) || Array.isArray(graphData.edges))) {
                 if (graphData.nodes.length > 0 || graphData.edges.length > 0) {
                    renderCytoscapeGraph(graphData); // Call rendering function
                 } else {
                     if (cyContainer) cyContainer.innerHTML = '<p>未找到相关图谱数据 (节点/边为空)。</p>';
                 }
            } else {
                 // Handle cases where response is OK but data format is wrong
                 console.warn("Received unexpected graph data format:", graphData);
                 if (cyContainer) cyContainer.innerHTML = '<p>返回的图谱数据格式不正确。</p>';
            }
         } else {
            // Handle graph fetch error
            const errorText = await graphResponse.text();
            if (cyContainer) cyContainer.innerHTML = `<p>加载图谱数据出错 ${graphResponse.status}: ${errorText}</p>`;
            console.error(`Graph fetch failed (${messageId}): ${errorText}`);
         }

    } catch (error) {
        console.error(`为消息 ${messageId} 获取细节时出错:`, error);
        // Display general errors if Promise.all fails or specific fetch errors weren't caught
        if (vectorContent.innerHTML.includes('加载中')) {
             vectorContent.innerHTML = `<p>加载向量细节失败: ${error.message}</p>`;
        }
         // Ensure cyContainer is updated, not cyTargetDiv which might be removed by error handling
         const cyTargetDiv = document.getElementById('cy');
        if (cyTargetDiv && cyTargetDiv.innerHTML.includes('加载中') && cyContainer) {
             cyContainer.innerHTML = `<p>加载图谱细节失败: ${error.message}</p>`;
        } else if (!cyTargetDiv && cyContainer && cyContainer.innerHTML.includes('加载中')) {
            cyContainer.innerHTML = `<p>加载图谱细节失败: ${error.message}</p>`;
        }
    }
}

function clearRetrievalResults(showLoading = false) {
    // Clears or sets loading state for vector and graph areas
    vectorContent.innerHTML = showLoading ? '<p>加载向量数据中...</p>' : '<p>点击历史记录查看详情。</p>';

    // Clear Cytoscape graph
    if (currentCytoscapeInstance) {
        currentCytoscapeInstance.destroy();
        currentCytoscapeInstance = null;
    }
     // Ensure the #cy div exists inside the container for placing text/graph
     if (cyContainer) {
        let cyTargetDiv = document.getElementById('cy');
        if (!cyTargetDiv) {
            cyTargetDiv = document.createElement('div');
            cyTargetDiv.id = 'cy';
            cyContainer.innerHTML = ''; // Clear container before adding div
            cyContainer.appendChild(cyTargetDiv);
        }
         cyTargetDiv.innerHTML = showLoading ? '<p>加载图谱数据中...</p>' : '<p>点击历史记录查看图谱。</p>';
     } else {
         console.error("Cytoscape container '#cy-container' not found.");
     }

}


function renderCytoscapeGraph(graphData) {
    // Renders the knowledge graph using Cytoscape.js
    console.log("Rendering graph data:", graphData);
    const cyTargetDiv = document.getElementById('cy'); // Target div inside the container

    if (!cyTargetDiv) {
         console.error("Cytoscape target div 'cy' not found in DOM.");
         // Optionally update the container with an error message
         if(cyContainer) cyContainer.innerHTML = '<p>无法渲染图谱：目标区域未找到。</p>';
         return; // Cannot render without the target div
    }
    cyTargetDiv.innerHTML = ''; // Clear previous graph or loading message

    // Destroy previous instance if it exists
    if (currentCytoscapeInstance) {
        currentCytoscapeInstance.destroy();
        currentCytoscapeInstance = null;
    }

    // Basic check for empty data to avoid Cytoscape errors
    if ((!graphData.nodes || graphData.nodes.length === 0) && (!graphData.edges || graphData.edges.length === 0)) {
        console.log("Graph data is empty, skipping Cytoscape rendering.");
        cyTargetDiv.innerHTML = '<p>无图谱数据可显示。</p>';
        return;
    }


    try {
        // Initialize Cytoscape
        currentCytoscapeInstance = cytoscape({
            container: cyTargetDiv,
            elements: { // Ensure data format is correct for Cytoscape
                nodes: graphData.nodes || [], // Expecting array of { data: { id: ..., ... } }
                edges: graphData.edges || []  // Expecting array of { data: { id: ..., source: ..., target: ...} }
            },
            style: [ // Your style definitions from the original code
                { selector: 'node', style: { 'background-color': 'data(color, "#888")', 'label': 'data(label)', 'width': 50, 'height': 50, 'font-size': '10px', 'text-valign': 'center', 'text-halign': 'center', 'color': '#000', 'text-outline-color': '#fff', 'text-outline-width': 1 } },
                { selector: 'edge', style: { 'line-color': 'data(color, "#ccc")', 'target-arrow-color': 'data(color, "#ccc")', 'curve-style': 'bezier', 'target-arrow-shape': 'triangle', 'label': 'data(label)', 'width': 2, 'font-size': '8px', 'text-rotation': 'autorotate', 'text-margin-y': -5, 'color': '#000', 'text-background-color': '#fff', 'text-background-opacity': 0.7, 'text-background-padding': '1px' } },
                { selector: '.highlighted-node', style: { 'background-color': '#FF5733', 'border-color': '#E84A27', 'border-width': 3, 'width': 60, 'height': 60, 'z-index': 10, 'shadow-blur': 10, 'shadow-color': '#FF5733', 'shadow-opacity': 0.8 } },
                { selector: '.highlighted-edge', style: { 'line-color': '#FF5733', 'target-arrow-color': '#FF5733', 'width': 4, 'z-index': 9, 'shadow-blur': 5, 'shadow-color': '#FF5733', 'shadow-opacity': 0.6 } }
            ],
            layout: { // Your layout options
                name: 'cose', fit: true, padding: 30, animate: true, animationDuration: 500, nodeRepulsion: 400000, idealEdgeLength: 100, nodeOverlap: 20
            }
        });

         // Apply highlighting based on data received from backend
        (graphData['highlighted-node'] || []).forEach(n => {
            if (n?.data?.id) {
                 try { currentCytoscapeInstance.getElementById(n.data.id).addClass('highlighted-node'); }
                 catch(e){ console.warn(`Highlight node error (ID: ${n.data.id}): ${e.message}`); } // Catch errors for missing elements
            }
         });
        (graphData['highlighted-edge'] || []).forEach(e => {
             if (e?.data?.id) {
                  try { currentCytoscapeInstance.getElementById(e.data.id).addClass('highlighted-edge'); }
                  catch(e){ console.warn(`Highlight edge error (ID: ${e.data.id}): ${e.message}`); } // Catch errors
             }
         });


        // Fit the graph to the viewport after layout is ready
        currentCytoscapeInstance.ready(() => {
             currentCytoscapeInstance.fit(null, 30);
        });
        console.log("Cytoscape graph rendered.");

    } catch (error) {
        console.error("Cytoscape rendering error:", error);
        cyTargetDiv.innerHTML = `<p>渲染图谱时出错: ${error.message}</p>`; // Display error in the div
        currentCytoscapeInstance = null; // Ensure instance is null on error
    }
}

// Helper to update internal answer state from backend response or history item
function updateAnswerStore(data) {
    // Prioritize specific fields if available (e.g., from /generate response)
    currentAnswers.query = data.query !== undefined ? data.query : currentAnswers.query;
    currentAnswers.vector = data.vectorAnswer !== undefined ? data.vectorAnswer : (data.vector_response !== undefined ? data.vector_response : "");
    currentAnswers.graph = data.graphAnswer !== undefined ? data.graphAnswer : (data.graph_response !== undefined ? data.graph_response : "");
    currentAnswers.hybrid = data.hybridAnswer !== undefined ? data.hybridAnswer : (data.hybrid_response !== undefined ? data.hybrid_response : "");
    console.log("Updated answer store:", currentAnswers);
}

// --- Other Utilities (Keep toggleResize, fetchAndDisplaySuggestions if used) ---
function toggleResize(iconElement, targetType = 'section') {
     const targetElement = iconElement.closest(targetType === 'section' ? '.section-box' : '.box');
     if (!targetElement) return;
     const isEnlarged = targetElement.classList.toggle('enlarged');
     iconElement.textContent = isEnlarged ? 'fullscreen_exit' : 'fullscreen';
      // Ensure Cytoscape graph resizes correctly if its container is resized
     if (targetElement.contains(cyContainer) || targetElement.id === 'cy-container') {
         if (currentCytoscapeInstance) {
             // Delay resize slightly to allow container animation/transition to finish
             setTimeout(() => {
                 currentCytoscapeInstance.resize();
                 currentCytoscapeInstance.fit(null, 30); // Re-fit after resize
             }, 300); // Adjust delay if needed
         }
     }
}
async function fetchAndDisplaySuggestions() {
    // Fetches and displays suggestions (ensure backend route is correct)
    adviceContent.innerHTML = "正在加载建议...";
    try {
        const response = await fetch('/get_suggestions'); // Ensure this endpoint works
        if (!response.ok) {
             const errorText = await response.text();
             throw new Error(`网络错误: ${response.status} ${errorText}`);
        }
        const data = await response.json();
        console.log("建议数据:", data);
        // Adapt rendering based on expected 'data' structure from backend
        if (data.advice) { // Assuming simple structure for now
             adviceContent.innerHTML = `
                 <h3>建议:</h3>
                 <p>${data.advice}</p>
                 `;
        } else {
             adviceContent.textContent = "未收到有效的建议数据。";
        }
    } catch (error) {
         console.error('获取建议时出错:', error);
         adviceContent.textContent = `无法加载建议: ${error.message}`;
    }
}

// --- Event Listeners ---
document.addEventListener('DOMContentLoaded', checkAuth); // Start auth check on page load
ragSelect.addEventListener("change", displaySelectedAnswer); // Update answer display on mode change
dim1Select.addEventListener('change', updateDatasetSelection); // Dataset dimension dropdowns
dim2Select.addEventListener('change', updateDatasetSelection);
dim3Select.addEventListener('change', updateDatasetSelection);

// Add listener for the logout button
if (logoutButton) {
    logoutButton.addEventListener('click', handleLogout);
}

// --- Initial call is now triggered by checkAuth -> initializeDemo ---