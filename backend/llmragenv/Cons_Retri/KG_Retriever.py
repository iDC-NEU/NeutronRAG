'''
Author: fzb fzb0316@163.com
Date: 2024-09-19 08:48:47
LastEditors: lpz 1565561624@qq.com
LastEditTime: 2025-03-27 10:40:43
FilePath: /RAGWebUi_demo/llmragenv/Retriever/retriever_graph.py
Description: 这是默认设置,请设置`customMade`, 打开koroFileHeader查看配置 进行设置: https://github.com/OBKoro1/koro1FileHeader/wiki/%E9%85%8D%E7%BD%AE
'''
'''

    这个文件用来根据用户输入来抽取实体、转换向量数据等操作数据库操作前、后的处理工作
    也包括prompt设置、让大模型生成知识图谱等操作
    也可以直接操作数据库

'''
# from icecream import ic
from llmragenv.LLM.llm_base import LLMBase
from database.graph.graph_database import GraphDatabase
import numpy as np
from llmragenv.Cons_Retri.Embedding_Model import EmbeddingEnv

import cupy as cp


keyword_extract_prompt = (
    # "A question is provided below. Given the question, extract up to {max_keywords} "
    # "keywords from the text. Focus on extracting the keywords that we can use "
    # "to best lookup answers to the question. Avoid stopwords.\n"
    # "Note, result should be in the following comma-separated format: 'KEYWORDS: <keywords>'\n"
    # # "Only response the results, do not say any word or explain.\n"
    # "---------------------\n"
    # "question: {question}\n"
    # "---------------------\n"
    # # "KEYWORDS: "
    "A question is provided below. Given the question, extract up to {max_keywords} "
    "keywords from the text. Focus on extracting the keywords that we can use "
    "to best lookup answers to the question. Avoid stopwords.\n"
    "Note, result should be in the following comma-separated format, and start with KEYWORDS:'\n"
    "Only response the results, do not say any word or explain.\n"
    "---------------------\n"
    "question: {question}\n"
    "---------------------\n"
)



llama_synonym_expand_prompt = (
    # "Generate synonyms or possible form of keywords up to {max_keywords} in total, "
    # "considering possible cases of capitalization, pluralization, common expressions, etc.\n"
    # "Provide all synonyms of keywords in comma-separated format: 'SYNONYMS: <synonyms>'\n"
    # # "Note, result should be in one-line with only one 'SYNONYMS: ' prefix\n"
    # # "Note, result should be in the following comma-separated format: 'SYNONYMS: <synonyms>\n"
    # # "Only response the results, do not say any word or explain.\n"
    # "Note, result should be in one-line, only response the results, do not say any word or explain.\n"
    # "---------------------\n"
    # "KEYWORDS: {question}\n"
    # "---------------------\n"
    # # "SYNONYMS: "
    "Generate synonyms or possible form of keywords up to {max_keywords} in total, "
    "considering possible cases of capitalization, pluralization, common expressions, etc.\n"
    "Provide all synonyms of keywords in comma-separated format: 'SYNONYMS: <synonyms>'\n"
    # "Note, result should be in one-line with only one 'SYNONYMS: ' prefix\n"
    # "Note, result should be in the following comma-separated format: 'SYNONYMS: <synonyms>\n"
    # "Only response the results, do not say any word or explain.\n"
    "Note, result should be in one-line, only response the results, do not say any word or explain.\n"
    "---------------------\n"
    "KEYWORDS: {question}\n"
    "---------------------\n")


embed_model = None

def get_text_embeddings(texts, step=400):
    global embed_model
    if not embed_model:
        embed_model = EmbeddingEnv(embed_name="BAAI/bge-small-en-v1.5",
                                   embed_batch_size=10)

    all_embeddings = []
    n_text = len(texts)
    for start in range(0, n_text, step):
        input_texts = texts[start:min(start + step, n_text)]
        embeddings = embed_model.get_embeddings(input_texts)

        all_embeddings += embeddings
        
    return all_embeddings

def get_text_embedding(text):
    global embed_model
    if not embed_model:
        embed_model = EmbeddingEnv(embed_name="BAAI/bge-small-en-v1.5",
                                   embed_batch_size=10)
    embedding = embed_model.get_embedding(text)
    return embedding

def cosine_similarity_cp(
    embeddings1,
    embeddings2,
) -> float:
    embeddings1_gpu = cp.asarray(embeddings1)
    embeddings2_gpu = cp.asarray(embeddings2)

    product = cp.dot(embeddings1_gpu, embeddings2_gpu.T)

    norm1 = cp.linalg.norm(embeddings1_gpu, axis=1, keepdims=True)
    norm2 = cp.linalg.norm(embeddings2_gpu, axis=1, keepdims=True)

    norm_product = cp.dot(norm1, norm2.T)

    cosine_similarities = product / norm_product

    return cp.asnumpy(cosine_similarities)

class RetrieverGraph(object):
    def __init__(self,llm:LLMBase, graphdb : GraphDatabase):
        self.graph_database = graphdb
        self._llm = llm
        
        self.triplet2id = self.graph_database.triplet2id
        self.triplet_embeddings = self.graph_database.triplet_embeddings

    def extract_keyword(self, question, max_keywords=5):
        prompt = keyword_extract_prompt.format(question=question, max_keywords=max_keywords)
        print(prompt)
        
        # 获取 LLM 的 response
        # if self._llm.__class__.__name__ == "OllamaClient":
        #     response = self._llm.chat_with_ai(prompt, info = "keyword")
        # else:
        #     response = self._llm.chat_with_ai(prompt)
        response = self._llm.chat_with_ai(prompt)
        
        # 处理 response，去掉 "KEYWORDS:" 前缀
        if response.startswith("KEYWORDS:"):
            response = response[len("KEYWORDS:"):].strip()  # 去掉前缀并去除多余的空格
        # ic(response)
        
        # 按逗号分割，并去除空格，转为小写
        keywords = [keyword.strip().lower() for keyword in response.split(",")]
        # keywords = ["'东北大学'", "'Ral'"]
        capitalized_keywords= [keyword.replace("'", '') for keyword in keywords]
        # ic(capitalized_keywords)

        # 只将每个关键词的第一个字母大写
        capitalized_keywords = [keyword.capitalize() for keyword in capitalized_keywords]
        capitalized_keywords.append("Apple")
        
        # ic(capitalized_keywords)
        # print(f"capitalized_keywords: {capitalized_keywords}")

        return capitalized_keywords

    def retrieve_2hop(self, question, pruning = None, build_node = False):
        self.pruning = pruning

        keywords = self.extract_keyword(question)
        print("keywords:",keywords)
        query_results = {}

        if pruning:
            rel_map = self.graph_database.get_rel_map(entities=keywords, limit=1000000)
        else:
            rel_map = self.graph_database.get_rel_map(entities=keywords)

        clean_rel_map = self.graph_database.clean_rel_map(rel_map)

        query_results.update(clean_rel_map)

        knowledge_sequence = self.graph_database.get_knowledge_sequence(query_results)

        if knowledge_sequence == []:
            return knowledge_sequence

        if self.pruning:
            pruning_knowledge_sequence, pruning_knowledge_dict = self.postprocess(question, knowledge_sequence)
            
            if build_node:
                self.nodes = self.graph_database.build_nodes(pruning_knowledge_sequence,
                                pruning_knowledge_dict)
        else:
            pruning_knowledge_sequence = knowledge_sequence
            if build_node:       
                self.nodes = self.graph_database.build_nodes(knowledge_sequence, rel_map)

        return pruning_knowledge_sequence
    

    def retrieve_2hop_with_keywords(self, question, keywords = [], pruning = None, build_node = False):
        self.pruning = pruning

        query_results = {}

        if pruning:
            rel_map = self.graph_database.get_rel_map(entities=keywords, limit=1000000)
        else:
            rel_map = self.graph_database.get_rel_map(entities=keywords)

        clean_rel_map = self.graph_database.clean_rel_map(rel_map)

        query_results.update(clean_rel_map)

        knowledge_sequence = self.graph_database.get_knowledge_sequence(query_results)

        if knowledge_sequence == []:
            return knowledge_sequence

        if self.pruning:
            pruning_knowledge_sequence, pruning_knowledge_dict = self.postprocess(question, knowledge_sequence)
            
            if build_node:
                self.nodes = self.graph_database.build_nodes(pruning_knowledge_sequence,
                                pruning_knowledge_dict)
        else:
            pruning_knowledge_sequence = knowledge_sequence
            if build_node:       
                self.nodes = self.graph_database.build_nodes(knowledge_sequence, rel_map)
                
        return pruning_knowledge_sequence
    

    def get_nodes(self):
        return self.nodes

    def postprocess(self, question, knowledge_sequence):
        if len(knowledge_sequence) == 0:
            return []
        
        kg_triplets = self.graph_database.kg_seqs_to_triplets(knowledge_sequence)
        kg_triplets = [' '.join(triplet) for triplet in kg_triplets]

        embedding_idxs = [
            self.triplet2id[triplet] for triplet in kg_triplets
            if triplet in self.triplet2id
        ]
        
        embeddings = self.triplet_embeddings[embedding_idxs]

        sorted_all_rel_scores = self.semantic_pruning_triplets(
            question,
            kg_triplets,
            rel_embeddings=embeddings,
            topk=self.pruning)

        pruning_knowledge_sequence = [rel for rel, _ in sorted_all_rel_scores]
        pruning_knowledge_dict = {"pruning": pruning_knowledge_sequence}

        return pruning_knowledge_sequence, pruning_knowledge_dict


    def semantic_pruning_triplets(self, question,
                              triplets,
                              rel_embeddings=None,
                              topk=30):
        question_embed = np.array(get_text_embedding(question)).reshape(1, -1)

        if rel_embeddings is None:
            rel_embeddings = get_text_embeddings(triplets)

        if len(rel_embeddings) == 1:
            rel_embeddings = np.array(rel_embeddings).reshape(1, -1)
        else:
            rel_embeddings = np.array(rel_embeddings)

        similarity_cp = cosine_similarity_cp(question_embed, rel_embeddings)[0]

        similarity = similarity_cp

        all_rel_scores = [(rel, score)
                        for rel, score in zip(triplets, similarity.tolist())]
        sorted_all_rel_scores = sorted(all_rel_scores,
                                    key=lambda x: x[1],
                                    reverse=True)

        return sorted_all_rel_scores[:topk]


