'''
Author: lpz 1565561624@qq.com
Date: 2025-07-30 11:49:21
LastEditors: lpz 1565561624@qq.com
LastEditTime: 2025-07-30 14:35:19
FilePath: /lipz/NeutronRAG/NeutronRAG/backend/schedule/request.py
Description: 这是默认设置,请设置`customMade`, 打开koroFileHeader查看配置 进行设置: https://github.com/OBKoro1/koro1FileHeader/wiki/%E9%85%8D%E7%BD%AE
'''
import time


WAITING = "waiting"
PROCESSING = "processing"
FINISHED = "finished"

class Request:
    def __init__(
        self,
        user_id: str,
        query_id: str,
        model_name:str,
        query: str,
        dataset_name:str,
        dataset_path:str,
        vector_retrieval: str,
        graph_retrieval: str,
        hybrid_retrieval:str,
        answer:str,
        state: str = None,
        timestamp: float = None,
        top_k=5,
        k_hop=2,
        keywords=5,
        pruning=False,
        item_data = {}
       
    ):
        self.user_id = user_id                  # 用户 ID
        self.query_id = query_id
        self.query = query                      # 查询内容
        self.vector_retrieval = vector_retrieval  # 是否启用向量检索
        self.graph_retrieval = graph_retrieval    # 是否启用图检索
        self.hybrid_retrieval = hybrid_retrieval
        self.state = state                      # 当前状态（如 init / processed / failed）
        self.timestamp = timestamp or time.time()  # 时间戳，默认为当前时间
        self.dataset_name = dataset_name
        self.top_k = top_k
        self.k_hop = k_hop
        self.keywords = keywords
        self.pruning = pruning
        self.model_name = model_name
        self.dataset_path = dataset_path
        self.vector_response = ""
        self.graph_response = ""
        self.hybrid_response = ""
        self.answer = answer
        self.v_flag = False
        self.g_flag = False
        self.h_flag = False
        self.item_data = item_data
        
    def to_dict(self) -> dict:
        return {
            "user_id": self.user_id,
            "query": self.query,
            "vector_retrieval": self.vector_retrieval,
            "graph_retrieval": self.graph_retrieval,
            "state": self.state,
            "timestamp": self.timestamp,
        }

    def set_response(self,response):
        self.response = response
        self.state = FINISHED

    def __repr__(self):
        return f"Request({self.to_dict()})"
    