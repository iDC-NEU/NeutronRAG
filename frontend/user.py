'''
Author: lpz 1565561624@qq.com
Date: 2025-02-09 18:31:22
LastEditors: lpz 1565561624@qq.com
LastEditTime: 2025-04-29 12:45:10
FilePath: /NeutronRAG/NeutronRAG/frontend/user.py
Description: 这是默认设置,请设置`customMade`, 打开koroFileHeader查看配置 进行设置: https://github.com/OBKoro1/koro1FileHeader/wiki/%E9%85%8D%E7%BD%AE
'''
import re
import mysql.connector
from config.config import db_chat_config 
from datetime import datetime

class User:
    def __init__(self, username: str, email: str, phone: str, password: str):
        self.username = username
        self.email = email
        self.phone = phone
        self.password = password
        self.user_id = None  # 初始化 user_id
        self.conn = mysql.connector.connect(**db_chat_config)  # 使用导入的配置创建连接  
        self.cursor = self.conn.cursor() 

    def _is_valid_email(self) -> bool:
        return re.match(r"[^@]+@[^@]+\.[^@]+", self.email) is not None

    def _is_valid_phone(self) -> bool:
        return re.match(r"^\d{11}$", self.phone) is not None

    def register(self, confirm_password: str) -> bool:
        if not self._is_valid_email():
            raise ValueError("邮箱格式不正确！")
        if not self._is_valid_phone():
            raise ValueError("手机号码格式不正确！")
        if self.password != confirm_password:
            raise ValueError("两次密码不匹配！")
        
        # 检查用户名是否已存在
        self.cursor.execute('SELECT COUNT(*) FROM user WHERE username = %s', (self.username,))
        if self.cursor.fetchone()[0] > 0:
            raise ValueError("用户名已存在！")
    
        # 检查邮箱是否已存在
        self.cursor.execute('SELECT COUNT(*) FROM user WHERE email = %s', (self.email,))
        if self.cursor.fetchone()[0] > 0:
            raise ValueError("邮箱已存在！")
    
        # 检查手机号是否已存在
        self.cursor.execute('SELECT COUNT(*) FROM user WHERE phone = %s', (self.phone,))
        if self.cursor.fetchone()[0] > 0:
            raise ValueError("手机号码已存在！")
        #插入数据
        self.cursor.execute('''
            INSERT INTO user (username, email, phone, password) 
            VALUES (%s, %s, %s, %s)
        ''', (self.username, self.email, self.phone, self.password))
        self.conn.commit()
        return True

    def login(self, password: str) -> bool:
        self.cursor.execute('SELECT password FROM user WHERE username = %s', (self.username,))
        row = self.cursor.fetchone()
        if row is None:  # 用户不存在
            raise ValueError("用户不存在！")
        if row[0] != password:  # 密码不正确
            raise ValueError("密码错误！")
        return True
    
    def get_user_id(self) -> int:
        self.cursor.execute('SELECT id FROM user WHERE username = %s', (self.username,))
        return self.cursor.fetchone()[0]

    def message_memory(self, session_id: int, user_input: str):
        self.cursor.execute('''
            INSERT INTO chat_messages (session_id, sender_type, message_text, send_time)
            VALUES (%d, %s, %s, %s)
        ''', (session_id, 'user', user_input, datetime.now()))
        self.conn.commit()
        return True

    def close_connection(self):
        self.cursor.close()
        self.conn.close()
