'''
Author: fzb fzb0316@163.com
Date: 2024-09-15 17:15:53
LastEditors: lpz 1565561624@qq.com
LastEditTime: 2025-07-30 16:36:48
FilePath: /RAGWebUi_demo/config/config.py
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
        root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
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
 
db_config = {  
    'host': 'localhost',  
    'user': 'root',  
    'password': 'a123456',  
    'database': 'chat'  
}

try:
    conn = mysql.connector.connect(**db_config)
    print("数据库连接成功！")
    conn.close()
except mysql.connector.Error as err:
    print(f"数据库连接失败: {err}")

    
# if __name__ == "__main__":
#     print(get_app_root())
