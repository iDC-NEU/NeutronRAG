document.getElementById("login-form").addEventListener("submit", function (e) {
    e.preventDefault(); // 阻止表单的默认提交行为，该行为会导致页面重新加载

    const username = document.getElementById("username").value;
    const password = document.getElementById("password").value;

    // 基本的前端验证
    if (!username || !password) {
        alert("用户名和密码是必需的！");
        return;
    }

    const data = {
        username: username,
        password: password
    };

    // 使用 Fetch API 将登录数据提交到后端的 /api/login 路由
    fetch('/api/login', { // 确保这与 app.py 中的 POST 路由匹配
        method: 'POST',
        headers: {
            'Content-Type': 'application/json',
            'Accept': 'application/json' // 表明我们期望返回 JSON
        },
        body: JSON.stringify(data) // 将 JS 对象转换为 JSON 字符串
    })
    .then(response => {
        // 检查 HTTP 响应状态是否表示成功 (例如 200 OK)
        if (response.ok) {
            return response.json(); // 如果成功，解析 JSON 主体
        } else {
            // 如果响应状态不是 OK (例如 400, 401, 500 等)
            // 改进的错误处理逻辑：
            return response.text().then(text => { // 1. 首先获取响应的文本内容
                let errorMessage;
                try {
                    const errData = JSON.parse(text); // 2. 尝试将文本解析为 JSON
                    if (errData && errData.error) {
                        errorMessage = errData.error; // 3. 如果解析成功且包含 error 字段，则使用它
                                                     //    例如："无效的用户名或密码"
                    } else {
                        // 如果 JSON 解析成功但结构不符合预期，或 error 字段不存在
                        console.warn("服务器返回了JSON错误，但格式不符合预期:", errData);
                        errorMessage = `服务器认证错误，但返回的具体信息格式不正确 (状态码: ${response.status})。`;
                        if (text) { // 为了调试，附加部分原始响应
                            errorMessage += ` 原始响应内容片段: "${text.substring(0, 100)}${text.length > 100 ? '...' : ''}"`;
                        }
                    }
                } catch (e) {
                    // 4. 如果文本无法解析为 JSON (说明服务器可能没有返回 JSON，或者返回的JSON无效)
                    console.error("无法将服务器的错误响应解析为JSON:", e);
                    errorMessage = `登录请求失败，状态码: ${response.status}。`;
                    if (text) { // 为了调试，附加部分原始响应
                        errorMessage += ` 服务器原始响应内容片段: "${text.substring(0, 100)}${text.length > 100 ? '...' : ''}"`;
                    } else {
                        errorMessage += " 服务器未返回响应内容。";
                    }
                }
                throw new Error(errorMessage); // 5. 抛出包含具体或更详细信息的错误
            });
        }
    })
    .then(data => {
        // 这个代码块只有在 response.ok 为 true 且 response.json() 成功时才会执行
        if (data.message) { // 后端发送了成功消息
            // alert(data.message); // 可选：显示成功消息的 alert
            console.log("Login successful:", data.message);
            // 登录成功后重定向到主应用页面
            window.location.href = '/'; // 重定向到根路径 (通常是主页或仪表盘)
        } else {
            // 如果后端成功响应 (2xx) 但没有 message 字段 (不太可能，但作为防御性编程)
            // 这种情况可能表示后端API设计上的不一致，但也应处理
            console.warn("登录成功，但服务器未返回明确的成功消息对象。");
            window.location.href = '/'; // 仍尝试跳转
        }
    })
    .catch(error => {
        // 捕获来自 fetch 本身的网络错误，或从 .then 链中抛出的任何错误
        console.error('Login Error:', error); // 在控制台打印详细错误对象
        alert(`登录失败: ${error.message}`); //直接显示 Error 对象的 message 属性给用户
                                         // 如果上面的错误处理逻辑正确，这里应该能显示具体的错误原因
    });
});