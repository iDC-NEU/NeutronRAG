'''
Author: lpz 1565561624@qq.com
Date: 2025-02-09 18:31:22
LastEditors: lpz 1565561624@qq.com
LastEditTime: 2025-05-07 20:02:08
FilePath: /lipz/NeutronRAG/NeutronRAG/frontend/config/config.py
Description: 这是默认设置,请设置`customMade`, 打开koroFileHeader查看配置 进行设置: https://github.com/OBKoro1/koro1FileHeader/wiki/%E9%85%8D%E7%BD%AE
'''

import threading
from functools import lru_cache

import yaml
import os


class Config(object):
    __instance = None
    __lock = threading.Lock()

    def __init__(self):
        self._config = None

    @classmethod
    def get_instance(cls):
        with cls.__lock:
            if cls.__instance is None:
                cls.__instance = cls._load_config()
            return cls.__instance

    @classmethod
    def _load_config(cls):
        instance = Config()
        root = os.getcwd()
        env = "local"
        with open(os.path.join(root, "config", f"config-{env}.yaml"), "r", encoding="utf-8") as f:
            setattr(instance, "_config", yaml.load(f, Loader=yaml.FullLoader))

        return instance

    @lru_cache(maxsize=128)
    def get_with_nested_params(self, *params):
        assert self._config is not None, "please load config first"
        conf = self._config
        for param in params:
            if param in conf:
                conf = conf[param]
            else:
                raise KeyError(f"{param} not found in config")

        return conf


import mysql.connector
import os

# MySQL 基本配置信息（未连接特定数据库）
db_config = {
    'host': 'localhost',
    'port': 3307,
    'user': 'root',
    'password': 'a123456'
}


sql_file_path = '../frontend/user.sql'

try:
    conn = mysql.connector.connect(**db_config)
    cursor = conn.cursor()
    print("成功连接到 MySQL 服务器！")
    

    if not os.path.exists(sql_file_path):
        raise FileNotFoundError(f"SQL 文件不存在: {sql_file_path}")
    with open(sql_file_path, 'r', encoding='utf-8') as f:
        sql_commands = f.read()
    for result in cursor.execute(sql_commands, multi=True):
        if result.with_rows:
            result.fetchall()
    print("数据库和数据表创建完成！")
    
    # 关闭连接，重新连接到特定数据库
    conn.close()
    db_chat_config = db_config.copy()
    db_chat_config['database'] = 'chat'  # 加上 database
    conn = mysql.connector.connect(**db_chat_config)
    cursor = conn.cursor()
    print("成功连接到 chat 数据库！")
    conn.close() 
    
except mysql.connector.Error as err:
    print(f"MySQL错误: {err}")
except Exception as e:
    print(f"其他错误: {e}")
finally:
    if 'cursor' in locals():
        cursor.close()
    if 'conn' in locals():
        conn.close()

# if __name__ == "__main__":
#     print(get_app_root())
