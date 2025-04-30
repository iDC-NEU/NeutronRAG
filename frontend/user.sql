CREATE DATABASE IF NOT EXISTS chat;
USE chat;

CREATE TABLE IF NOT EXISTS `user`  (
    `id` int NOT NULL AUTO_INCREMENT COMMENT 'Primary Key',
    `create_time` datetime DEFAULT NULL COMMENT 'Create Time',
    `username` varchar(50) NOT NULL,
    `email` varchar(100) NOT NULL,
    `phone` varchar(15) NOT NULL,
    `password` varchar(255) NOT NULL,
    PRIMARY KEY (`id`),
    UNIQUE KEY `email` (`email`),
    UNIQUE KEY `phone` (`phone`)
) ENGINE = InnoDB AUTO_INCREMENT = 9 DEFAULT CHARSET = utf8mb4 COLLATE = utf8mb4_0900_ai_ci;


/* CREATE TABLE IF NOT EXISTS `chat_sessions` (
    session_id INT AUTO_INCREMENT PRIMARY KEY,
    user_id INT NOT NULL,
    FOREIGN KEY (user_id) REFERENCES user(id)
);

CREATE TABLE IF NOT EXISTS `chat_messages` (
    session_id INT NOT NULL,
    sender_type ENUM('user', 'model') NOT NULL,
    message_text TEXT NOT NULL,
    send_time TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (session_id) REFERENCES chat_sessions(session_id)
); */

-- password: my-secret-pw