'''
Author: lpz 1565561624@qq.com
Date: 2025-03-19 20:28:13
LastEditors: lpz 1565561624@qq.com
LastEditTime: 2025-03-27 10:34:48
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
                 strategy="Union",
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


    # 为了不浪费 chat_vector 和 chat_graph的检索结果
    def hybrid_chat(self):
        retrieval_text = self.chat_vector.retrieval_result()
        retrieval_graph = self.chat_graph.retrieval_result()
        retrieval_result = ""
        if self.strategy == "Union":
            retrieval_result = retrieval_text+retrieval_graph
        elif self.strategy == "Intersection":
        




#TODO XINBO 这个函数我想的是后端实现历史信息的读取与生成，首先我们根据 self.path_name找到要生成的路径（若没有则自动创建）， 如果有 则看里面是否有内容，如果有内容，看是继续生成还是重新生成（is_continue参数）
# 这个路径里还是串行生成（这个目前卡有限，之后再看是否加并行优化）三个json文件吧（对于一个item，先生成vector，再生成graph。然后根据strategy生成hybrid（hybrid_chat我还没写之后我再加上））对于生成的格式(graph和vector的既要有生成指标也要有检索指标)

# 至于Evaluator我大概后天写完，当需要用到评估值时，写自己写一个值代替吧


    def history_chat(self,is_continue:bool):



        return

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

        
    



if __name__ == "__main__":
    chat = Demo_chat(model_name="llama3:8b",dataset="rgb")
    print(chat.chat_test())
    chat.close()





