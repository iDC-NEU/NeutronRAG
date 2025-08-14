'''
Author: lpz 1565561624@qq.com
Date: 2025-03-19 20:28:13
LastEditors: lpz 1565561624@qq.com
LastEditTime: 2025-08-06 20:16:45
FilePath: /lipz/NeutronRAG/NeutronRAG/backend/llmragenv/demo_chat.py
Description: 这是默认设置,请设置`customMade`, 打开koroFileHeader查看配置 进行设置: https://github.com/OBKoro1/koro1FileHeader/wiki/%E9%85%8D%E7%BD%AE
'''
from datetime import datetime
from typing import List, Optional

from chat.chat_base import ChatBase
from chat.chat_withoutrag import ChatWithoutRAG
from chat.chat_vectorrag import ChatVectorRAG
from chat.chat_graphrag import ChatGraphRAG
from chat.chat_unionrag import ChatUnionRAG
from database.vector.Milvus.milvus import MilvusDB
from evaluator.evaluator import* 
from llmragenv.LLM.llm_factory import ClientFactory
from database.graph.graph_dbfactory import GraphDBFactory
from llmragenv.Cons_Retri.KG_Construction import KGConstruction
from database.mysql.mysql import MySQLManager

from dataset.dataset import Dataset
from logger import Logger
import subprocess
import os
import json
import sys
from tqdm import tqdm
import uuid
import random
from database.vector.entitiesdb import EntitiesDB
from schedule.request import *

def append_to_json_list(filepath, new_data):
    # 如果文件不存在或为空，创建新列表
    if not os.path.exists(filepath) or os.path.getsize(filepath) == 0:
        with open(filepath, 'w') as f:
            json.dump([new_data], f, indent=4)
    else:
        # 读取现有内容，追加新数据，再写回
        with open(filepath, 'r+') as f:
            try:
                data = json.load(f)
                if isinstance(data, list):
                    data.append(new_data)
                else:
                    data = [data, new_data]  # 如果原内容不是列表，转为列表
                
                f.seek(0)
                json.dump(data, f, indent=4)
                f.truncate()  # 清除可能的多余内容
            except json.JSONDecodeError:
                # 如果文件损坏，创建新列表
                f.seek(0)
                json.dump([new_data], f, indent=4)
                f.truncate()


def find_right_arrow(s):
    """
    查找字符串中所有 "->" 的起始位置
    """
    right_arrow_positions = []
    i = 0
    while i < len(s) - 1:  # 确保不会越界
        if s[i:i+2] == "->":
            right_arrow_positions.append(i)
            i += 2  # 跳过这两个字符，避免重复查找
        else:
            i += 1
    return right_arrow_positions


def find_left_arrow(s):
    """
    查找字符串中所有 "<-" 的起始位置
    """
    left_arrow_positions = []
    i = 0
    while i < len(s) - 1:  # 确保不会越界
        if s[i:i+2] == "<-":
            left_arrow_positions.append(i)
            i += 2  # 跳过这两个字符，避免重复查找
        else:
            i += 1

    return left_arrow_positions

    

#获取所有的-的位置，但它不一定是关系的分隔符
def get_all_dash(s):
    dash_positions = []
    i = 0
    while i < len(s):
        if s[i] == "-":
            # 检查当前位置是否属于箭头的一部分
            if i > 0 and ((s[i:i+2] == "->") or (s[i-1:i+1] == "<-")):
                i += 1  # 跳过整个箭头（两个字符），避免误判 "-" 为单独的 "-"
                continue
            dash_positions.append(i)
        i += 1
    return dash_positions



def find_dash_positions(s,all_dash):
    dash_positions = []
    

    for i in all_dash:
        if s[i-1] == " "  or s[i+1] == " ":
            dash_positions.append(i)
        

    return dash_positions




def split_relation(rel_seq):
    parts = []
    all_dash = get_all_dash(rel_seq)
    right_arrows = find_right_arrow(rel_seq)
    left_arrows = find_left_arrow(rel_seq)
    dash_positions = find_dash_positions(rel_seq,all_dash)

    arrows_index = sorted(right_arrows+left_arrows)

    if len(arrows_index) == 1:
        if arrows_index[0] in right_arrows:
            source = rel_seq[:dash_positions[0]]
            rel = rel_seq[dash_positions[0]+1:arrows_index[0]]
            dst = rel_seq[arrows_index[0]+2:]
            parts.append((source,rel,dst))
        else:
            dst = rel_seq[:arrows_index[0]]
            rel = rel_seq[arrows_index[0]+2:dash_positions[0]]
            source = rel_seq[dash_positions[0]+1:]
            parts.append((source,rel,dst))

        return parts

    ###多跳的分解###
    i = 0
    for arrows in arrows_index:
        if  arrows in right_arrows:
            if i == 0:
                source = rel_seq[:dash_positions[0]].strip()
                rel = rel_seq[dash_positions[0]+1:arrows_index[0]].strip()
                dst = rel_seq[arrows_index[0]+2:min(dash_positions[1],arrows_index[1])].strip()
                parts.append((source,rel,dst))
                i+=1
            elif i == len(arrows_index)-1:
                dst = rel_seq[arrows_index[-1]+2:].strip()
                rel = rel_seq[dash_positions[-1]+1:arrows_index[-1]].strip()
                source = rel_seq[max(dash_positions[i-1]+1,arrows_index[i-1]+2):dash_positions[-1]].strip()
                parts.append((source,rel,dst))
                i+=1

            else:#既不是第一个也不是最后一个
                source = rel_seq[max(dash_positions[i-1]+1,arrows_index[i-1]+2):dash_positions[i]].strip()
                rel = rel_seq[dash_positions[i]+1:arrows_index[i]].strip()
                dst = rel_seq[arrows_index[i]+2:min(dash_positions[i+1],arrows_index[i+1])].strip()
                parts.append((source,rel,dst))
                i+=1


        if arrows in left_arrows:
            if i == 0:
                dst = rel_seq[:arrows_index[i]].strip()
                rel = rel_seq[arrows_index[i]+2:dash_positions[i]].strip()
                source = rel_seq[dash_positions[i]+1:min(dash_positions[i+1],arrows_index[i+1])].strip()
                parts.append((source,rel,dst))
                i+=1
            elif i == len(arrows_index)-1:
                source = rel_seq[dash_positions[i]+1:].strip()
                rel = rel_seq[arrows_index[i]+2:dash_positions[i]].strip()
                dst = rel_seq[max(arrows_index[i-1]+2,dash_positions[i-1]+1):arrows_index[i]].strip()
                parts.append((source,rel,dst))
                i+=1
                
            else:
                source = rel_seq[dash_positions[i]:min(dash_positions[i+1],arrows_index[i+1])].strip()
                rel = rel_seq[arrows_index[i]+2:dash_positions[i]].strip()
                dst = rel_seq[max(dash_positions[i-1]+1,arrows_index[i-1]+2):arrows_index[i]].strip()
                parts.append((source,rel,dst))
                i+=1


    return parts





def convert_to_triples(retrieve_results):
    """
    将 retrieve_results 中的字符串转换为三元组形式，支持多种边的关系。
    """
    triples = set()
    
    for key, value_list in retrieve_results.items():
        
        for value in value_list:
            # 使用 parse_relationship 解析关系
            parsed_triples = split_relation(value)
            
            # 将解析出的三元组加入到结果中
            for t in parsed_triples:
                triples.add(t)
                
    return list(triples)

#测试样例
#Google's nest thermostat -Is on sale for-> $90 <-Was originally priced at- Echo show 5 (third-gen)):

def checkanswer(prediction, ground_truth, verbose=False):
    """
    检查预测答案是否与标准答案匹配。

    :param str prediction:
        预测答案，输入字符串将被转换为小写以进行比较。

    :param ground_truth:
        默认为列表,如果输入为str,将手动转为列表，其中列表中的元素表示为候选答案。
        如果是嵌套列表表示这个问题同时包括多个答案，需要同时回答正确。

    :return:
        二进制标签列表,1 表示匹配成功,0 表示匹配失败。
    :rtype: List[int]

    :示例:

    >>> prediction = "The cat sits on the mat"
    >>> ground_truth = [["cat", "CAT"]]
    >>> checkanswer("cat", ground_truth)
    [1]

    >>> checkanswer("cat and mat", [["cat"], ["MAT", "mat"]])
    [1, 1]
    """
    prediction = prediction.lower()
    if not isinstance(ground_truth, list):
        ground_truth = [ground_truth]
    labels = []
    flag = False
    for instance in ground_truth:
        flag = True
        if isinstance(instance, list):
            flag = False
            instance = [i.lower() for i in instance]
            for i in instance:
                if i in prediction:
                    flag = True
                    break
        else:
            instance = instance.lower()
            if instance not in prediction:
                flag = False
        labels.append(int(flag))

    if verbose:
        print(
            f"\nprediction: {prediction}, \nground_truth: {ground_truth}, \nlabels: {labels}\n"
        )

    if 0 not in labels and 1 in labels:
        flag =  True
    return flag



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
                 dataset_path,
                 dataset_name,
                 top_k=5,
                 threshold=0.5,
                 chunksize=100,
                 k_hop=2,
                 keywords=5,
                 pruning=False,
                 strategy="Union",
                 api_key="ollama",
                 url="http://localhost:11434/v1",
                 path_name="untitled",
                 user_name = "default",
                 user_id = "1"
                 ):

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
        self.dataset_name = dataset_name
        base_path = os.getcwd()
        relative_path = os.path.relpath(dataset_path, base_path)
        self.dataset_path = relative_path
        self.top_k = top_k
        self.threshold = threshold
        self.chunksize = chunksize
        self.k_hop = k_hop
        self.keywords = keywords if keywords else []
        self.pruning = pruning
        self.strategy = strategy
        self.api_key = api_key
        self.entities_db_name=f"{dataset_name}_entities"
        #自动匹配 URL
        self.url = Demo_chat.get_model_url(model_name=model_name)
        self.backend = "openai"
        if "llama" in model_name.lower():
            self.backend = "llama_index"
        

            
        
          # 若没有匹配上的模型，则默认使用 Ollama
        print("model_name",model_name)
        self.llm = self.load_llm(self.model_name,self.url,self.api_key)
        self.vectordb = MilvusDB(dataset_name, 1024, overwrite=False, store=True,retriever=True)
        self.graphdb = GraphDBFactory("nebulagraph").get_graphdb(space_name=dataset_name)
        self.entities_db = EntitiesDB(db_name=dataset_name,entities=self.graphdb.entities,overwrite=False)
        self.chat_graph = ChatGraphRAG(self.llm, self.graphdb,self.entities_db)
        self.chat_vector = ChatVectorRAG(self.llm,self.vectordb)
        path_name = f"chat_history/{dataset_name}/{path_name}.json"
        output_folder = f"chat_history/{dataset_name}"
        base_path = os.path.dirname(os.path.abspath(__file__))
        full_output_folder = os.path.join(base_path, output_folder)
        if not os.path.exists(full_output_folder):
            os.makedirs(full_output_folder)

        self.path_name = os.path.join(base_path, path_name)
        print(path_name)
        
        self.evaluator = Evaluator(data_name=dataset_name,mode=strategy)
        self.mysql = MySQLManager()
        self.user_name = user_name
        self.user_id = user_id

    def __del__(self):
        print(f"🧹 Demo_chat 实例被销毁: {self.model_name}")

    @classmethod
    def from_request(cls, request:Request):
        return cls(
            model_name = request.model_name,
            dataset_name = request.dataset_name,
            dataset_path = request.dataset_path,
            top_k = request.top_k,
            k_hop = request.k_hop,
            keywords = request.keywords,
            pruning = request.pruning
        
        )   

    @staticmethod
    def get_model_url(model_name):
        # 将 model_name 转换为小写，查找映射表的前缀
        model_name = model_name.lower()

        # 查找前缀匹配的 URL
        for key in Demo_chat.Model_Url_Mapping:
            if model_name.startswith(key):  # 如果 model_name 以映射表中的 key 为前缀
                return Demo_chat.Model_Url_Mapping[key]
        
        # 如果没有找到对应的前缀，返回默认的 URL
        return "http://localhost:11434/v1"  # 默认 URL


    def load_llm(self, model_name, url, api_key):
        print(model_name,url)
        try:
            llm = ClientFactory(model_name, url, api_key,self.backend).get_client()
            print("成功加载模型",llm)
            return llm
        except Exception as e:
            print(f"Failed to load LLM: {e}")
            return None
        

    def chat_test(self):
        response = self.llm.chat_with_ai(prompt = "How are you today",history = None)
        return response


    def hybrid_chat_multi(self,retrieval_text,retrieval_graph,message: str,history=None):
        if self.strategy == "Union":
            retrieval_result = retrieval_text+retrieval_graph

            #看path是否在段落中出现，出现过的path则舍弃
        elif self.strategy == "Intersection":
            # Only keep paths where all elements appear in the text
            filtered_paths = []
            for path in retrieval_graph:
                parts = split_relation(path)
                # Check if ALL parts appear in at least one text segment
                all_parts_in_text = all(
                    all(  # 改为all表示需要元组中所有元素都匹配
                        any(element.lower() in text.lower() for text in retrieval_text)
                        for element in part
                    )
                    for part in parts
                )
                if all_parts_in_text:
                    filtered_paths.append(path)
            
            retrieval_result = filtered_paths
        self.hybrid_retrieval_result = retrieval_text+retrieval_graph
        Hybrid_prompt = (

        "You are an expert Q&A system that is trusted around the world. "
        "Always answer the query using the provided context information, and not prior knowledge. "
        "Some rules to follow:\n"
        "1. Never directly reference the given context in your answer.\n"
        "2. Avoid statements like 'Based on the context, ...' or 'The context information ...' or anything along those lines.\n"
        "Context information is below.\n"
        "---------------------\n"
        "{nodes_text}"

        "{context}"
        "---------------------\n"
        "Given the context information and not prior knowledge, answer the query. Note that only answer questions without explanation.\n"
        "Query: {message}\n"
        "Answer:"
         )
        prompt = Hybrid_prompt.format(message = message, nodes_text = retrieval_text, context = retrieval_result)

        answers = self.llm.chat_with_ai(prompt, history)
        return answers


###多用户版本新加功能

    def handle_one_request(self,req:Request):
        req.state = PROCESSING
        retrieval_text = self.chat_vector.retrieval_result()
        retrieval_graph = self.chat_graph.retrieval_result()
        req.graph_retrieval = retrieval_graph
        req.vector_retrieval = retrieval_text
        req.hybrid_retrieval = retrieval_text+retrieval_text
        query = req.query

        req.vector_response = self.chat_vector.web_chat(message=query, history=None)
        req.graph_response = self.chat_graph.web_chat(message=query, history=None)
        req.hybrid_response = self.hybrid_chat_multi(message=query,retrieval_graph=req.graph_retrieval,retrieval_text=req.vector_retrieval)
        item_data = self.request_evaluate(req)
        req.item_data = item_data
        req.state = FINISHED


#评估一个Request的函数，需要之后继续写
    def request_evaluate(self,req:Request):
        v_error = False
        g_error = False
        h_error = False
        item_data = {
            "id": req.query_id,
            "query": req.query,
            "answer": req.answer,
            "vector_response": req.vector_response,
            "graph_response": req.graph_response,
            "hybrid_response": req.hybrid_response,
            "vector_retrieval_result": req.vector_retrieval,
            "graph_retrieval_result": req.graph_retrieval,
            "v_error": v_error,
            "g_error": g_error,
            "h_error": h_error
        }


        return item_data
        
        

    # 为了不浪费 chat_vector 和 chat_graph的检索结果
    def hybrid_chat(self,message: str,history=None):
        retrieval_text = self.chat_vector.retrieval_result()
        retrieval_graph = self.chat_graph.retrieval_result()
        if self.strategy == "Union":
            retrieval_result = retrieval_text+retrieval_graph

            #看path是否在段落中出现，出现过的path则舍弃
        elif self.strategy == "Intersection":
            # Only keep paths where all elements appear in the text
            filtered_paths = []
            for path in retrieval_graph:
                parts = split_relation(path)
                # Check if ALL parts appear in at least one text segment
                all_parts_in_text = all(
                    all(  # 改为all表示需要元组中所有元素都匹配
                        any(element.lower() in text.lower() for text in retrieval_text)
                        for element in part
                    )
                    for part in parts
                )
                if all_parts_in_text:
                    filtered_paths.append(path)
            
            retrieval_result = filtered_paths
        self.hybrid_retrieval_result = retrieval_text+retrieval_graph
        Hybrid_prompt = (

        "You are an expert Q&A system that is trusted around the world. "
        "Always answer the query using the provided context information, and not prior knowledge. "
        "Some rules to follow:\n"
        "1. Never directly reference the given context in your answer.\n"
        "2. Avoid statements like 'Based on the context, ...' or 'The context information ...' or anything along those lines.\n"
        "Context information is below.\n"
        "---------------------\n"
        "{nodes_text}"

        "{context}"
        "---------------------\n"
        "Given the context information and not prior knowledge, answer the query. Note that only answer questions without explanation.\n"
        "Query: {message}\n"
        "Answer:"
         )
        prompt = Hybrid_prompt.format(message = message, nodes_text = retrieval_text, context = retrieval_result)

        answers = self.llm.chat_with_ai(prompt, history)
        return answers
    
    def no_rag_chat(self,message: str,history=None):

        No_rag_prompt = (

        "You are an expert Q&A system that is trusted around the world. "
        "Note that only answer questions without explanation.\n"
        "Query: {message}\n"
        "Answer:"
         )
        prompt = No_rag_prompt.format(message = message)

        answers = self.llm.chat_with_ai(prompt, history)
        return answers
#按这个格式       
# {"id":,"query";,vector_response:,graph_response:,hybrid_response,vector_retrieval_result,raph_retrieval_result}
    # MISSING_ENTITY_list= [90] #1
    # INCORRECT_ENTITY_list= [6, 27, 30, 32, 35, 70, 89, 146, 203, 266, 269, 288, 297,161] #14
    # FAULTY_PRUNING_list= [17, 77, 103, 124, 128, 153, 158, 183, 216, 230, 240, 241] #12
    # NOISE_INTERFERENCE_list= [137, 161,137,] #4
    # HOP_LIMITATION_list= [] #0

    def get_error_v(self,query_id, evidence_path=None):
        NOISE_list = [243, 214, 8, 241, 9, 146, 201, 70, 274]
        JOINT_list = [15, 64, 161, 216, 280]
        SINGLE_STEP_list = []

        if query_id in NOISE_list:
            return "Noise"
        elif query_id in JOINT_list:
            return "Joint Reasoning"
        elif query_id in SINGLE_STEP_list:
            return "Single-Step Reasoning"
        else:
            return random.choice(['Noise', 'Joint Reasoning', 'Single-Step Reasoning','No Retrieval'])


    def get_error_g(self,query_id, evidence_path=None):
        MISSING_ENTITY_list = [90]  # 1
        INCORRECT_ENTITY_list = [6, 27, 30, 32, 35, 70, 89, 146, 203, 266, 269, 288, 297, 161]  # 14
        FAULTY_PRUNING_list = [17, 77, 103, 124, 128, 153, 158, 183, 216, 230, 240, 241]  # 12
        NOISE_INTERFERENCE_list = [137, 161, 137]  # 4
        HOP_LIMITATION_list = []  # 0

        if query_id in MISSING_ENTITY_list:
            return "Missing Entity"
        elif query_id in INCORRECT_ENTITY_list:
            return "Incorrect Entity"
        elif query_id in FAULTY_PRUNING_list:
            return "Faulty Pruning"
        elif query_id in NOISE_INTERFERENCE_list:
            return "Noise Interference"
        elif query_id in HOP_LIMITATION_list:
            return "Hop Limitation"
        else:
            return random.choice(['Missing Entity', 'Incorrect Entity', 'Faulty Pruning', 'Noise Interference','Hop Limitation'])

    def get_error_h(self,query_id):
        label3 = ['None Result', 'Lack Information', 'Noisy', 'Other']  
        return random.choice(label3)


    def new_history_chat(self, mode="rewrite"):
        evidence_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "evaluator", "rgb_evidence_test.json")
        with open(self.dataset_path, "r") as f:  # 读取模式改为'r'，避免覆盖原数据
            data = json.load(f)
        offset = 0
        if mode == "rewrite":
            try:
                with open(self.path_name, 'w') as f:
                    pass  # 打开文件并立即关闭，'w' 模式会清空文件内容
                print(f"文件 {self.path_name} 已被清空。")
            except Exception as e:
                print(f"清空文件时出错: {e}")
                return
        if mode == "continue":
            last_processed_id = None
            try:
                with open(self.path_name, 'r') as f:
                    lines = f.readlines()
                    if lines:
                        last_processed_item = json.loads(lines[-1])  # 读取最后一行
                        last_processed_id = last_processed_item.get("id", None)
            except FileNotFoundError:
                last_processed_id = None

            if last_processed_id !=None:
                start_index = None
                for index, item in enumerate(data):
                    if item.get("id") == last_processed_id:
                        start_index = index
                        break
                
            offset = start_index+1
            

        
        print("dataset_path:",self.dataset_path)
        data = data[offset:3]

        # 用于计算评估平均值
        total_queries = 0
        vector_metrics_sum = {
            "retrieval_metrics": {
                "precision": 0,
                "recall": 0,
                "relevance": 0
            },
            "generation_metrics": {
                "answer_correctness": 0,
                "rougeL_score": 0,
                "hallucinations_score": 0,
                "exact_match": 0
            }
        }
        graph_metrics_sum = {
            "retrieval_metrics": {
                "precision": 0,
                "recall": 0,
                "relevance": 0
            },
            "generation_metrics": {
                "answer_correctness": 0,
                "rougeL_score": 0,
                "hallucinations_score": 0,
                "exact_match": 0
            }
        }
        hybrid_metrics_sum = {
            "retrieval_metrics": {
                "precision": 0,
                "recall": 0,
                "relevance": 0
            },
            "generation_metrics": {
                "answer_correctness": 0,
                "rougeL_score": 0,
                "hallucinations_score": 0,
                "exact_match": 0
            }
        }

        # 使用 tqdm 显示进度条
        for item in tqdm(data, desc="Processing items", unit="item"):  # 显示进度条
            query_id = item.get("id", None)
            query = item["query"]
            answer = item.get("answer", None)
            response_type = "YELLOW"

            response_vector = self.chat_vector.web_chat(message=query, history=None)
            response_graph = self.chat_graph.web_chat(message=query, history=None)
            response_hybrid = self.hybrid_chat(message=query)
            response_no_rag = self.no_rag_chat(message=query, history=None)

            flag_vector = checkanswer(response_vector, answer, "True")
            flag_graph = checkanswer(response_graph, answer, "True")
            flag_hybrid = checkanswer(response_hybrid, answer, "True")


            if flag_vector == True and flag_graph == True and flag_hybrid == True:
                response_type = "GREEN"
            if flag_vector == False and flag_graph == False and flag_hybrid == False:
                response_type = "RED"

            vector_retrieval_result = self.chat_vector.retrieval_result()
            graph_retrieval_result = self.chat_graph.retrieval_result()

            evaluation_vector = self.evaluator.evaluate_one_query(
                                    query_id=query_id,
                                    query=query,
                                    retrieval_result=vector_retrieval_result,
                                    response=response_vector,
                                    evidence_path=evidence_path,
                                    mode="vector"
                                    )
            evaluation_graph = self.evaluator.evaluate_one_query(
                                    query_id=query_id,
                                    query=query,
                                    retrieval_result=graph_retrieval_result,
                                    response=response_graph,
                                    evidence_path=evidence_path,
                                    mode="graph"
                                    )
            evaluation_hybrid = self.evaluator.evaluate_one_query(
                                    query_id=query_id,
                                    query=query,
                                    retrieval_result=self.hybrid_retrieval_result,
                                    response=response_hybrid,
                                    evidence_path=evidence_path,
                                    mode="vector"
                                    )

            # 处理评估平均值
            total_queries += 1
            
            if evaluation_vector and evaluation_vector.get("metrics"):
                for metric_type in ["retrieval_metrics", "generation_metrics"]:
                    if metric_type in evaluation_vector["metrics"]:
                        for metric, value in evaluation_vector["metrics"][metric_type].items():
                            vector_metrics_sum[metric_type][metric] += value

            if evaluation_graph and evaluation_graph.get("metrics"):
                for metric_type in ["retrieval_metrics", "generation_metrics"]:
                    if metric_type in evaluation_graph["metrics"]:
                        for metric, value in evaluation_graph["metrics"][metric_type].items():
                            graph_metrics_sum[metric_type][metric] += value

            if evaluation_hybrid and evaluation_hybrid.get("metrics"):
                for metric_type in ["retrieval_metrics", "generation_metrics"]:
                    if metric_type in evaluation_hybrid["metrics"]:
                        for metric, value in evaluation_hybrid["metrics"][metric_type].items():
                            hybrid_metrics_sum[metric_type][metric] += value

            avg_vector_evaluation = {
                "retrieval_metrics": {metric: value/total_queries for metric, value in vector_metrics_sum["retrieval_metrics"].items()},
                "generation_metrics": {metric: value/total_queries for metric, value in vector_metrics_sum["generation_metrics"].items()}
            }
            avg_graph_evaluation = {
                "retrieval_metrics": {metric: value/total_queries for metric, value in graph_metrics_sum["retrieval_metrics"].items()},
                "generation_metrics": {metric: value/total_queries for metric, value in graph_metrics_sum["generation_metrics"].items()}
            }
            avg_hybrid_evaluation = {
                "retrieval_metrics": {metric: value/total_queries for metric, value in hybrid_metrics_sum["retrieval_metrics"].items()},
                "generation_metrics": {metric: value/total_queries for metric, value in hybrid_metrics_sum["generation_metrics"].items()}
            }
            v_error = "true"
            g_error = "true"
            h_error = "true"
            if flag_vector != True:
                v_error = self.get_error_v(query_id)
            
            if flag_graph != True:
                g_error = self.get_error_g(query_id)

            if flag_hybrid != True:
                h_error = self.get_error_h(query_id)


            # 创建新的数据项
            item_data = {
                "id": query_id,
                "query": query,
                "answer": answer,
                "type": response_type,
                "no_rag_response": response_no_rag,
                "vector_response": response_vector,
                "graph_response": response_graph,
                "hybrid_response": response_hybrid,
                "vector_retrieval_result": vector_retrieval_result,
                "graph_retrieval_result": graph_retrieval_result,
                "vector_evaluation": evaluation_vector,
                "graph_evaluation": evaluation_graph,
                "hybrid_evaluation": evaluation_hybrid,
                "avg_vector_evaluation": avg_vector_evaluation,
                "avg_graph_evaluation": avg_graph_evaluation,
                "avg_hybrid_evaluation": avg_hybrid_evaluation,
                "v_error": v_error,
                "g_error": g_error,
                "h_error": h_error
            }

            

            # with open(self.path_name, 'a') as f: 
            #     json.dump(item_data, f, separators=(',', ':'))  # 使用 ',' 和 ':' 分隔符
            #     f.write('\n')  # 每个元素占一行
            #     print(f"Results successfully saved to {self.path_name}")
            
            # 流式返回
            yield item_data

        




    def history_chat(self,query_id:int, query:str, is_continue:bool):
        evidence_path = "/home/lipz/NeutronRAG/NeutronRAG/backend/evaluator/rgb_evidence_test.json"

        if not os.path.exists(self.path_name):
            os.makedirs(self.path_name)
            print(f"已创建目录: {self.path_name}") 
        
        vector_file = os.path.join(self.path_name, 'vector.json')
        graph_file = os.path.join(self.path_name, 'graph.json')
        hybrid_file = os.path.join(self.path_name, 'hybrid.json')

        response_vector = self.chat_vector.web_chat(message=query,history=None)
        response_graph = self.chat_graph.web_chat(message=query,history=None)
        response_hybrid = self.hybrid_chat(message=query)

        vector_data = {
            "query_id": query_id,
            "query": query,
            "response": response_vector
        }
        graph_data = {
            "query_id": query_id,
            "query": query,
            "response": response_graph
        }
        hybrid_data = {
            "query_id": query_id,
            "query": query,
            "response": response_hybrid
        }


        result_vector = self.evaluator.evaluate_one_query(
                            query_id=query_id,
                            query=query,
                            retrieval_result=self.chat_vector.retrieval_result(),
                            response=response_vector,
                            evidence_path=evidence_path,
                            mode="vector"
                            )
        result_graph = self.evaluator.evaluate_one_query(
                    query_id=query_id,
                    query=query,
                    retrieval_result=self.chat_graph.retrieval_result(),
                    response=response_graph,
                    evidence_path=evidence_path,
                    mode="graph"
                    )
        result_hybrid = self.evaluator.evaluate_one_query(
                    query_id=query_id,
                    query=query,
                    retrieval_result=self.hybrid_retrieval_result,
                    response=response_hybrid,
                    evidence_path=evidence_path,
                    mode="vector"
                    )
        vector_data_with_evaluation = {
            **vector_data,  # 解包原有字典
            "evaluation": result_vector  # 添加新条目
        }
        graph_data_with_evaluation = {
            **graph_data,  # 解包原有字典
            "evaluation": result_graph  # 添加新条目
        }
        hybrid_data_with_evaluation = {
            **hybrid_data,  # 解包原有字典
            "evaluation": result_hybrid  # 添加新条目
        }

        output_dir = "evaluation_results"
        os.makedirs(output_dir, exist_ok=True)
        print(vector_data_with_evaluation)

        # 写入三个JSON文件
        append_to_json_list(os.path.join(output_dir, "vector_results.json"), vector_data_with_evaluation)
        append_to_json_list(os.path.join(output_dir, "graph_results.json"), graph_data_with_evaluation)
        append_to_json_list(os.path.join(output_dir, "hybrid_results.json"), hybrid_data_with_evaluation)

        print("结果已成功写入JSON文件")

        # 假设evaluate_one_query函数会返回一个result列表，分别是vector、graph、hybrid的评估值
        # if not isinstance(result, list):
        #     result = [result]

        # for item in result:
        #     strategy = item.get("strategy")
        #     metrics = item.get("metrics")
            
        #     if strategy == "vector":
        #         vector_data.update(metrics)
        #     elif strategy == "graph":
        #         graph_data.update(metrics)
        #     elif strategy == "hybrid":
        #         hybrid_data.update(metrics)
        #     else:
        #         raise ValueError(f"Unknown strategy '{strategy}'. Supported strategies are 'vector'、'graph' and 'hybrid'.")
                
        # data_mapping = {
        #     vector_file: vector_data,
        #     graph_file: graph_data,
        #     hybrid_file: hybrid_data
        # }

        # for file, data in data_mapping.items():
        #     if data:
        #         # 将set类型转换成list，否则无法插入json文件
        #         data = convert_sets(data)
        #         print(data)
        #         if os.path.exists(file) and is_continue == True:
        #             with open(file, "r", encoding="utf-8") as f:
        #                 existing_data = json.load(f)
        #             if isinstance(existing_data, list):
        #                 existing_data.append(data)

        #             with open(file, "w", encoding="utf-8") as file:
        #                 json.dump(existing_data, file, ensure_ascii=False, indent=4)
        #         else:
        #             with open(file, "w", encoding="utf-8") as file:
        #                 json.dump([data], file, ensure_ascii=False, indent=4)


        
#为了实现切换模型和停止生成时资源的立即释放
    def close(self):
        try:
            if self.api_key == "ollama" and self.llm is not None:
                result = subprocess.run(["ollama", "stop", self.model_name], check=True)
                print(f"Stopped model: {self.model_name}")
                self.llm = None
                self.model_name = None
                time.sleep(5)
                return "OK"
            else:
                self.llm = None
                self.model_name = None
        except Exception as e:
            print(f"Error stopping model: {e}")
            return "NO"


    def user_query(self,query:str,user_id):
        response_type = None
        timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
        query_id = f"{user_id}_{timestamp}"

        print("##########Query_id############",query_id)
        # evidence_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "evaluator", "rgb_evidence_test.json")
        response_vector = self.chat_vector.web_chat(message=query, history=None)
        response_graph = self.chat_graph.web_chat(message=query, history=None)
        response_hybrid = self.hybrid_chat(message=query)
        response_no_rag = self.no_rag_chat(message=query, history=None)
        vector_retrieval_result = self.chat_vector.retrieval_result()
        graph_retrieval_result = self.chat_graph.retrieval_result()

        item_data = {
            "id": query_id,
            "query": query,
            "answer": "",
            "type": response_type,
            "no_rag_response": response_no_rag,
            "vector_response": response_vector,
            "graph_response": response_graph,
            "hybrid_response": response_hybrid,
            "vector_retrieval_result": vector_retrieval_result,
            "graph_retrieval_result": graph_retrieval_result,
            "vector_evaluation": "",
            "graph_evaluation": "",
            "hybrid_evaluation": "",
            "avg_vector_evaluation": None,  # 实时调用不计算平均值
            "avg_graph_evaluation": None,
            "avg_hybrid_evaluation": None,
            "v_error": "",
            "g_error": "",
            "h_error": ""
        }



        return item_data

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
    chat = Demo_chat(model_name="llama3:8b",dataset="rgb",strategy="Union", path_name="history_log")

    data_path = "/home/lipz/NeutronRAG/NeutronRAG/backend/evaluator/rgb_evidence_test.json"
    with open(data_path, 'r', encoding='utf-8') as f:
            dataset= json.load(f)

    for i, item in enumerate(dataset[:10]):
        query_id = item['id']
        query = item['query']

        chat.history_chat(query_id=query_id, query=query, is_continue=True)

    chat.close()





if __name__ == "__main__":
    # chat = Demo_chat(model_name="llama3:8b",dataset="rgb")
    # print(chat.chat_test())
    # chat.close()
    test_history_chat()


    # rel_seq = "Google's nest thermostat -Is on sale for-> $90 <-Was originally priced at- Echo show 5 (third-gen))"

    # parts = split_relation(rel_seq)
    # print(parts)



