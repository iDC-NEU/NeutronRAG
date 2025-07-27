'''
Author: fzb0316 fzb0316@163.com
Date: 2024-09-21 19:23:18
LastEditors: lpz 1565561624@qq.com
LastEditTime: 2025-07-27 16:14:10
FilePath: /BigModel/RAGWebUi_demo/llmragenv/Cons_Retri/Embedding_Model.py
Description: 这是默认设置,请设置`customMade`, 打开koroFileHeader查看配置 进行设置: https://github.com/OBKoro1/koro1FileHeader/wiki/%E9%85%8D%E7%BD%AE
'''
import numpy as np
import requests
from transformers import AutoTokenizer, AutoModel
from llama_index.embeddings.huggingface import HuggingFaceEmbedding



EMBEDD_DIMS = {
    "BAAI/bge-large-en-v1.5": 1024,
    "BAAI/bge-base-en-v1.5": 768,
    "BAAI/bge-small-en-v1.5": 384,
    "text-embedding-ada-002": 1536,
    "qllama/bge-large-en-v1.5:f16": 1024,
}
class EmbeddingEnv:

    def __init__(self,
                 embed_name="BAAI/bge-large-en-v1.5",
                 embed_batch_size=20,
                 device="cuda:0"):
        self.embed_name = embed_name
        self.embed_batch_size = embed_batch_size

        assert embed_name in EMBEDD_DIMS
        self.dim = EMBEDD_DIMS[embed_name]

        if 'BAAI' in embed_name:
            print(f"use huggingface embedding {embed_name}")
            self.embed_model = HuggingFaceEmbedding(
                model_name=embed_name,
                embed_batch_size=embed_batch_size,
                device=device)
        # else:
        #     print(f"use openai embedding {embed_name}")
        #     self.embed_model = OpenAIEmbedding(
        #         model=embed_name, embed_batch_size=embed_batch_size)

        # Settings.embed_model = self.embed_model
        # print_text(
        #     f"EmbeddingEnv: embed_name {embed_name}, embed_batch_size {self.embed_batch_size}, dim {self.dim}\n",
        #     color='red')

    def __str__(self):
        return f"{self.embed_name} {self.embed_batch_size}"

    def get_embedding(self, query_str):
        embedding = self.embed_model._get_text_embedding(query_str)
        return embedding

    def get_embeddings(self, query_str_list):
        embeddings = self.embed_model._get_text_embeddings(query_str_list)
        return embeddings

    def calculate_similarity(self, query1, query2):
        # Cosine similarity
        embedding1 = self.embed_model.get_embedding(query1)
        embedding2 = self.embed_model.get_embedding(query2)
        dot_product = np.dot(embedding1, embedding2)
        norm1 = np.linalg.norm(embedding1)
        norm2 = np.linalg.norm(embedding2)
        similarity = round(dot_product / (norm1 * norm2), 6)
        return similarity

class OllamaEmbedding:
    def __init__(self, model_name="qllama/bge-large-en-v1.5:f16", embed_batch_size=20, device="cuda:0"):
        self.model_name = model_name
        self.embed_batch_size = embed_batch_size
        self.device = device  # 当前未使用，但可用于扩展

    def _get_text_embedding(self, text):
        """获取单条文本的 embedding"""
        url = "http://localhost:11434/api/embeddings"
        payload = {
            "model": self.model_name,
            "prompt": text
        }
        response = requests.post(url, json=payload)
        response.raise_for_status()
        return np.array(response.json()["embedding"])

    def _get_text_embeddings(self, texts):
        """获取多条文本的 embedding（批量）"""
        return [self._get_text_embedding(text) for text in texts]
 
    
class Ollama_EmbeddingEnv:
    def __init__(self,
                 embed_name="qllama/bge-large-en-v1.5:f16",
                 embed_batch_size=20,
                 device="cuda:0"):
        self.embed_name = embed_name
        self.embed_batch_size = embed_batch_size

        assert embed_name in EMBEDD_DIMS
        self.dim = EMBEDD_DIMS[embed_name]

        print(f"use Ollama embedding {embed_name}")
        self.embed_model = OllamaEmbedding(
            model_name=embed_name,
            embed_batch_size=embed_batch_size,
            device=device
        )

    def __str__(self):
        return f"{self.embed_name} {self.embed_batch_size}"

    def get_embedding(self, query_str):
        embedding = self.embed_model._get_text_embedding(query_str)
        return embedding.tolist()

    def get_embeddings(self, query_str_list):
        embeddings = self.embed_model._get_text_embeddings(query_str_list)
        return embeddings

    def calculate_similarity(self, query1, query2):
        # Cosine similarity
        embedding1 = self.embed_model.get_embedding(query1)
        embedding2 = self.embed_model.get_embedding(query2)
        dot_product = np.dot(embedding1, embedding2)
        norm1 = np.linalg.norm(embedding1)
        norm2 = np.linalg.norm(embedding2)
        similarity = round(dot_product / (norm1 * norm2), 6)
        return similarity

    
if __name__ == '__main__':
    Embed = EmbeddingEnv()
    a = Embed.get_embedding("Hello")
    print(a)
    env = Ollama_EmbeddingEnv()
    vec = env.get_embedding("hello world")
    print(vec)