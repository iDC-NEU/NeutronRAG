'''
Author: fzb0316 fzb0316@163.com
Date: 2024-09-21 19:23:18
LastEditors: lpz 1565561624@qq.com
LastEditTime: 2025-08-05 19:23:05
FilePath: /BigModel/RAGWebUi_demo/llmragenv/Cons_Retri/retriever_vector.py
Description: 这是默认设置,请设置`customMade`, 打开koroFileHeader查看配置 进行设置: https://github.com/OBKoro1/koro1FileHeader/wiki/%E9%85%8D%E7%BD%AE
'''

'''

    这个文件用来根据用户输入来抽取实体、转换向量数据等操作数据库操作前、后的处理工作
    也包括prompt设置、让大模型生成知识图谱等操作
    但这个文件不直接操作数据库

'''
from ast import List
from transformers import AutoTokenizer, AutoModel
import torch
from huggingface_hub import hf_hub_download
from llmragenv.LLM.llm_base import LLMBase
from database.vector.vector_database import VectorDatabase
from llama_index.core.utils import print_text
from llama_index.vector_stores.milvus import MilvusVectorStore
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core.schema import NodeWithScore, QueryBundle
from llama_index.core.retrievers import (

    # KnowledgeGraphRAGRetriever,
    VectorIndexRetriever)
from llama_index.core import (
    VectorStoreIndex,
    # SimpleDirectoryReader,
    # Document,
    StorageContext,
    load_index_from_storage,
)







class RetrieverVector(object):

    def __init__(self,llm:LLMBase, vectordb: VectorDatabase):
        self.vector_database = vectordb
        self._llm = llm
    def format_chunks(self,retrieval_results):
        formatted_string = ""
        for i, chunk in enumerate(retrieval_results, 1):
            formatted_string += f"Chunk{i}: {chunk}\n"
        return formatted_string
        

    def retrieve(self,question):
        retrieval_result = []
        embedding = self.vector_database.embed_model.get_embedding(question)
        nodes = self.vector_database.retrieve_nodes(question,embedding)
        for node in nodes:
            retrieval_result.append(node.text)

        # formatted_string = format_chunks(retrieval_result)
        # print(formatted_string)

        return retrieval_result
        