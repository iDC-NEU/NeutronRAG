document.getElementById("register-form").addEventListener("submit", function(e) {
    e.preventDefault();  // Prevent default form submission

    // Get form field values
    const username = document.getElementById("username").value;
    const email = document.getElementById("email").value;
    const phone = document.getElementById("phone").value;
    const password = document.getElementById("password").value;
    const confirmPassword = document.getElementById("confirm_password").value;

    // --- Basic Frontend Validations ---
    // Email format
    const emailRegex = /^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$/;
    if (!emailRegex.test(email)) {
        alert("请输入有效的邮箱地址！");
        return;
    }
    // Basic China phone number format check
    const phoneRegex = /^1\d{10}$/;
    if (!phoneRegex.test(phone)) {
        alert("请输入有效的11位中国大陆手机号！");
        return;
    }
    // Check if passwords match
    if (password !== confirmPassword) {
        alert("密码和确认密码不匹配！");
        return;
    }
    // Check if required fields are filled
    if (!username || !email || !phone || !password) {
        alert("所有字段均为必填项！");
        return;
    }
    // Add more checks if needed (e.g., password complexity)
    // --- End Validations ---

    // Prepare data object for submission
    const data = {
        username: username,
        email: email,
        phone: phone,
        password: password,
        confirm_password: confirmPassword // Send confirm_password for backend validation too (optional)
    };

    // Use Fetch API to send registration data to the backend's /register route
    fetch('/api/register', { // Ensure this matches the POST route in app.py
        method: 'POST',
        headers: {
            'Content-Type': 'application/json',
            'Accept': 'application/json' // Indicate we expect JSON back
        },
        body: JSON.stringify(data) // Convert JS object to JSON string
    })
    .then(response => {
         // Check if the HTTP response status indicates success (e.g., 201 Created)
         if (response.ok) {
             return response.json(); // Parse JSON body
         } else {
             // If response status is not OK, try to parse error JSON
             return response.json().then(errData => {
                 // Throw an error with the message from backend or a default one
                 throw new Error(errData.error || `注册失败，状态码: ${response.status}`);
             }).catch(() => {
                 // If parsing error JSON fails, throw generic error
                  throw new Error(`注册失败，状态码: ${response.status}`);
             });
         }
    })
    .then(data => {
        // This block executes only if the response was OK
        if (data.message) { // Backend sends a success message
            alert(data.message); // Show success message (e.g., "注册成功！请登录。")
            // Redirect to the login page after successful registration
            window.location.href = '/login';
        }
    })
    .catch(error => {
        // Catches errors from fetch or errors thrown from .then
        console.error('Registration Error:', error);
        alert(`注册失败: ${error.message}`); // Display error message to the user
    });
});