document.getElementById("register-form").addEventListener("submit", function(e) {
    e.preventDefault(); // 阻止表单默认提交行为

    // 获取表单字段的值
    const username = document.getElementById("username").value;
    const email = document.getElementById("email").value;
    const phone = document.getElementById("phone").value;
    const password = document.getElementById("password").value;
    const confirmPassword = document.getElementById("confirm_password").value;

    // --- 基本的前端验证 ---
    // 邮箱格式
    const emailRegex = /^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$/;
    if (!emailRegex.test(email)) {
        alert("请输入有效的邮箱地址！");
        return;
    }
    // 基本的中国大陆手机号格式检查
    const phoneRegex = /^1\d{10}$/;
    if (!phoneRegex.test(phone)) {
        alert("请输入有效的11位中国大陆手机号！");
        return;
    }
    // 检查密码是否匹配
    if (password !== confirmPassword) {
        alert("密码和确认密码不匹配！");
        return;
    }
    // 检查必填字段是否已填写
    if (!username || !email || !phone || !password) {
        alert("所有字段均为必填项！");
        return;
    }
    // 如果需要，可以添加更多检查 (例如密码复杂度)
    // --- 验证结束 ---

    // 准备要提交的数据对象
    const data = {
        username: username,
        email: email,
        phone: phone,
        password: password,
        confirm_password: confirmPassword // 也发送 confirm_password 以便后端再次验证 (可选)
    };

    // 使用 Fetch API 将注册数据发送到后端的 /api/register 路由
    fetch('/api/register', { // 确保这与 app.py 中的 POST 路由匹配
        method: 'POST',
        headers: {
            'Content-Type': 'application/json',
            'Accept': 'application/json' // 表明我们期望返回 JSON
        },
        body: JSON.stringify(data) // 将 JS 对象转换为 JSON 字符串
    })
    .then(response => {
        // 检查 HTTP 响应状态是否表示成功 (例如 201 Created)
        if (response.ok) {
            return response.json(); // 解析 JSON 主体
        } else {
            // 如果响应状态不是 OK (例如 400, 401, 500 等)
            // 改进的错误处理逻辑：
            return response.text().then(text => { // 1. 首先获取响应的文本内容
                let errorMessage;
                try {
                    const errData = JSON.parse(text); // 2. 尝试将文本解析为 JSON
                    if (errData && errData.error) {
                        errorMessage = errData.error; // 3. 如果解析成功且包含 error 字段，则使用它
                                                     //    例如："手机号已被注册"
                    } else {
                        // 如果 JSON 解析成功但结构不符合预期，或 error 字段不存在
                        console.warn("服务器返回了JSON错误，但格式不符合预期:", errData);
                        errorMessage = `服务器错误，但返回的具体信息格式不正确 (状态码: ${response.status})。`;
                        if (text) { // 为了调试，附加部分原始响应
                            errorMessage += ` 原始响应内容片段: "${text.substring(0, 100)}${text.length > 100 ? '...' : ''}"`;
                        }
                    }
                } catch (e) {
                    // 4. 如果文本无法解析为 JSON (说明服务器可能没有返回 JSON，或者返回的JSON无效)
                    console.error("无法将服务器的错误响应解析为JSON:", e);
                    errorMessage = `注册请求失败，状态码: ${response.status}。`;
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
            alert(data.message); // 显示成功消息 (例如 "注册成功！请登录。")
            // 注册成功后重定向到登录页面
            window.location.href = '/login';
        } else {
            // 如果后端成功响应 (2xx) 但没有 message 字段 (不太可能，但作为防御性编程)
            alert("注册操作已处理，但未收到明确的成功消息。");
            window.location.href = '/login'; // 仍尝试跳转
        }
    })
    .catch(error => {
        // 捕获来自 fetch 本身的错误，或从 .then 链中抛出的任何错误
        console.error('Registration Error:', error); // 在控制台打印详细错误对象
        alert(`${error.message}`); // 直接显示 Error对象的 message 属性给用户
                                     // 如果上面的错误处理逻辑正确，这里应该能显示具体的错误原因
    });
});