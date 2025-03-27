'''
Author: lpz 1565561624@qq.com
Date: 2025-03-19 20:28:13
LastEditors: lpz 1565561624@qq.com
LastEditTime: 2025-03-25 23:13:24
FilePath: /lipz/NeutronRAG/NeutronRAG/backend/llmragenv/demo_chat.py
Description: 这是默认设置,请设置`customMade`, 打开koroFileHeader查看配置 进行设置: https://github.com/OBKoro1/koro1FileHeader/wiki/%E9%85%8D%E7%BD%AE
'''
from typing import List, Optional

from chat.chat_base import ChatBase
from chat.chat_withoutrag import ChatWithoutRAG
from chat.chat_vectorrag import ChatVectorRAG
from chat.chat_graphrag import ChatGraphRAG
from chat.chat_unionrag import ChatUnionRAG
from database.vector.Milvus.milvus import MilvusDB
from llmragenv.LLM.llm_factory import ClientFactory
from database.graph.graph_dbfactory import GraphDBFactory
from llmragenv.Cons_Retri.KG_Construction import KGConstruction

from dataset.dataset import Dataset
from logger import Logger
import subprocess
import os
import json
import sys
home_dir = os.path.expanduser("~")
evaluator_path = os.path.join(home_dir, "NeutronRAG/backend/evaluator")
sys.path.append(evaluator_path)
from evaluator import Evaluator

class Demo_chat:

    # 模型到 URL 的映射表
    Model_Url_Mapping = {
        "zhipu": "https://open.bigmodel.cn/api/paas/v4",
        "moonshot": "https://api.moonshot.cn/v1",
        "baichuan": "https://api.baichuan-ai.com/v1",
        "qwen": "https://dashscope.aliyuncs.com/compatible-mode/v1",
        "lingyiwanwu": "https://api.lingyiwanwu.com/v1",
        "deepseek": "https://api.deepseek.com",
        "doubao": "https://ark.cn-beijing.volces.com/api/v3",
        "gpt": "https://api.aigc798.com/v1/",
        "llama": "http://localhost:11434/v1",  # 本地 Ollama
    }

    def __init__(self,
                 model_name,
                 dataset,
                 top_k=5,
                 threshold=0.5,
                 chunksize=100,
                 k_hop=2,
                 keywords=None,
                 pruning=False,
                 strategy="default",
                 api_key="ollama",
                 url="http://localhost:11434/v1",
                 path_name="untitled"):

        """
        初始化 Demo_chat 类。

        :param model_name: 使用的模型名称
        :param dataset: 语料库或数据集
        :param top_k: 选择前 k 个最佳答案
        :param threshold: 置信度阈值
        :param chunksize: 处理数据时的分块大小
        :param k_hop: k-hop 查询的步长（用于知识图谱）
        :param keywords: 关键词列表
        :param pruning: 是否进行剪枝优化
        :param strategy: 检索或生成的策略
        """
        self.model_name = model_name
        self.dataset = dataset
        self.top_k = top_k
        self.threshold = threshold
        self.chunksize = chunksize
        self.k_hop = k_hop
        self.keywords = keywords if keywords else []
        self.pruning = pruning
        self.strategy = strategy
        self.api_key = api_key
        #自动匹配 URL
        self.url = self.Model_Url_Mapping.get(model_name, "http://localhost:11434/v1")  # 若没有匹配上的模型，则默认使用 Ollama
        self.llm = self.load_llm(self.model_name,self.url,self.api_key)
        self.vectordb = MilvusDB(dataset, 1024, overwrite=False, store=True,retriever=True)
        self.graphdb = GraphDBFactory("nebulagraph").get_graphdb(space_name='rgb')
        self.chat_graph = ChatGraphRAG(self.llm, self.graphdb)
        self.chat_vector = ChatVectorRAG(self.llm,self.vectordb)
        self.path_name = path_name
        self.evaluator = Evaluator(data_name=dataset,mode=strategy)


        

    def load_llm(self, model_name, url, api_key):
        try:
            llm = ClientFactory(model_name, url, api_key).get_client()
            return llm
        except Exception as e:
            print(f"Failed to load LLM: {e}")
            return None
        

    def chat_test(self):
        response = self.llm.chat_with_ai(prompt = "How are you today",history = None)
        return response


    def history_chat(self,query_id:int, query:str, response:str, is_continue:bool, result):

        if not os.path.exists(self.path_name):
            os.makedirs(self.path_name)
            print(f"已创建目录: {self.path_name}") 
        
        vector_file = os.path.join(self.path_name, 'vector.json')
        graph_file = os.path.join(self.path_name, 'graph.json')
        hybrid_file = os.path.join(self.path_name, 'hybrid.json')

        vector_data = {
            "query_id": query_id,
            "query": query,
            "response": response
        }
        graph_data = {
            "query_id": query_id,
            "query": query,
            "response": response
        }
        hybrid_data = {
            "query_id": query_id,
            "query": query,
            "response": response
        }

        # 假设evaluate_one_query函数会返回一个result列表，分别是vector、graph、hybrid的评估值
        if not isinstance(result, list):
            result = [result]

        for item in result:
            strategy = item.get("strategy")
            metrics = item.get("metrics")
            
            if strategy == "vector":
                vector_data.update(metrics)
            elif strategy == "graph":
                graph_data.update(metrics)
            elif strategy == "hybrid":
                hybrid_data.update(metrics)
            else:
                raise ValueError(f"Unknown strategy '{strategy}'. Supported strategies are 'vector'、'graph' and 'hybrid'.")
                
        data_mapping = {
            vector_file: vector_data,
            graph_file: graph_data,
            hybrid_file: hybrid_data
        }

        for file, data in data_mapping.items():
            if data:
                # 将set类型转换成list，否则无法插入json文件
                data = convert_sets(data)
                print(data)
                if os.path.exists(file) and is_continue == True:
                    with open(file, "r", encoding="utf-8") as f:
                        existing_data = json.load(f)
                    if isinstance(existing_data, list):
                        existing_data.append(data)

                    with open(file, "w", encoding="utf-8") as file:
                        json.dump(existing_data, file, ensure_ascii=False, indent=4)
                else:
                    with open(file, "w", encoding="utf-8") as file:
                        json.dump([data], file, ensure_ascii=False, indent=4)

    # def hybrid_chat(self,strategy):

        
#为了实现切换模型和停止生成时资源的立即释放
    def close(self):
        if self.api_key == "ollama" and self.llm is not None:
            subprocess.run(["ollama", "stop", self.model_name])
            print(f"Stopped model: {self.model_name}")
            self.llm = None
            self.model_name = None
        else:
            self.llm = None
            self.model_name = None


# 将set类型转换成list
def convert_sets(obj):
    if isinstance(obj, set):
        return list(obj)
    elif isinstance(obj, dict):
        return {k: convert_sets(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_sets(v) for v in obj]
    else:
        return obj

# 测试history_chat函数
def test_history_chat():
    chat = Demo_chat(model_name="llama3:8b",dataset="rgb",strategy="vector", path_name="history_log")

    data_path = "/home/yangxb/NeutronRAG/backend/evaluator/rgb_evidence_test.json"
    with open(data_path, 'r', encoding='utf-8') as f:
            dataset= json.load(f)

    for i, item in enumerate(dataset[:10]):
        query_id = item['id']
        query = item['query']
        evidences = item['evidences']
        response = item['response']
        result = chat.evaluator.evaluate_one_query(
                                    query_id=query_id,
                                    query=query,
                                    retrieval_result=evidences,
                                    response=response,
                                    vector_evidence=data_path,
                                    graph_evidence=data_path
                                    )
        chat.history_chat(query_id=query_id, query=query, response=response, is_continue=True, result=result)

    chat.close()



if __name__ == "__main__":
    chat = Demo_chat(model_name="llama3:8b",dataset="rgb")
    print(chat.chat_test())
    chat.close()
    # test_history_chat()





