document.getElementById("login-form").addEventListener("submit", function (e) {
    e.preventDefault();  // Prevent default form submission which would cause a page reload

    const username = document.getElementById("username").value;
    const password = document.getElementById("password").value;

    // Basic frontend validation
    if (!username || !password) {
        alert("用户名和密码是必需的！");
        return;
    }

    const data = {
        username: username,
        password: password
    };

    // Use Fetch API to submit login data to the backend's /login route
    fetch('/api/login', { // Ensure this matches the POST route in app.py
        method: 'POST',
        headers: {
            'Content-Type': 'application/json',
            'Accept': 'application/json' // Indicate we expect JSON back
        },
        body: JSON.stringify(data) // Convert JS object to JSON string
    })
    .then(response => {
        // Check if the HTTP response status is OK (e.g., 200)
        if (response.ok) {
            return response.json(); // Parse JSON body if successful
        } else {
            // If response status is not OK, try to parse error JSON from backend
            // Need to return a rejected Promise to be caught by .catch()
            return response.json().then(errData => {
                // Throw an error with the message from backend, or a default one
                throw new Error(errData.error || `登录失败，状态码: ${response.status}`);
            }).catch(() => {
                 // If parsing error JSON fails, throw a generic error
                 throw new Error(`登录失败，状态码: ${response.status}`);
            });
        }
    })
    .then(data => {
        // This block executes only if the response was OK
        if (data.message) { // Backend sends a success message
            // alert(data.message); // Optional: Show success message alert
            console.log("Login successful");
            // Redirect to the main application page upon successful login
            window.location.href = '/'; // Redirect to root (demo page)
        }
        // No 'else' needed here, errors are handled by the .catch block
    })
    .catch(error => {
        // Catches errors from fetch itself or errors thrown from .then block
        console.error('Login Error:', error);
        alert(`登录失败: ${error.message}`); // Display error message to the user
    });
});