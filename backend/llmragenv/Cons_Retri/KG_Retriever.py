'''
Author: fzb fzb0316@163.com
Date: 2024-09-19 08:48:47
LastEditors: lpz 1565561624@qq.com
LastEditTime: 2025-08-06 16:09:42
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
from llmragenv.Cons_Retri.Embedding_Model import EmbeddingEnv,Ollama_EmbeddingEnv
from llmragenv.Cons_Retri.pruning import *
import cupy as cp
from database.vector.entitiesdb import EntitiesDB


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
        
        # self.triplet2id = self.graph_database.triplet2id
        # self.triplet_embeddings = self.graph_database.triplet_embeddings

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
        
        # ic(capitalized_keywords)
        # print(f"capitalized_keywords: {capitalized_keywords}")

        return capitalized_keywords

    def retrieve_2hop(self, question, pruning = True, build_node = False):
        self.pruning = pruning

        keywords = self.extract_keyword(question)
        print("keywords:",keywords)
        

        if pruning:
            rel_map = self.graph_database.get_rel_map(entities=keywords, limit=10)
            # print("##########rel_map##########",rel_map)
        else:
            rel_map = self.graph_database.get_rel_map(entities=keywords)

        clean_rel_map = self.graph_database.clean_rel_map(rel_map)
        print("############clean_rel_map#################",clean_rel_map)
        all_knowledge_sequence = []
        for triples in clean_rel_map.values():
            all_knowledge_sequence.extend(triples)
        pruning_knowledge_sequence = semantic_pruning(question=question,knowledge_sequence=all_knowledge_sequence)
        pruned_sequence_only = [triple for triple, score in pruning_knowledge_sequence]
        print("#############pruned_sequence_only################",pruned_sequence_only)

        return pruned_sequence_only
    

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
    

    ###修改剪枝策略##
    def postprocess_clean_rel_map(self, question,clean_rel_map):
        if len(clean_rel_map) == 0:
            return []


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


# 这个剪枝类只在EntitiesDB检索的时候使用
class Pruning:

    def __init__(
        self,
        device="cuda:0",
        batch_size=10,
        embed_model="qllama/bge-large-en-v1.5:f16",
        step=100,
    ):
        self.step = step
        self.embed_model = Ollama_EmbeddingEnv(embed_name=embed_model, embed_batch_size=batch_size, device=device)


    def get_text_embedding(self, text):
        embedding = self.embed_model.get_embedding(text)
        return embedding

    def get_text_embeddings(self, texts):
        all_embeddings = []
        n_text = len(texts)
        for start in range(0, n_text, self.step):
            input_texts = texts[start : min(start + self.step, n_text)]
            embeddings = self.embed_model.get_embeddings(input_texts)
            all_embeddings += embeddings
        return all_embeddings

    def cosine_similarity_cp(
        self,
        embeddings1,
        embeddings2,
    ) -> float:
        # with cp.cuda.Device(0):
        #     arr1 = cp.array([1, 2, 3])
        #     print(cp.cuda.runtime.getDevice())  # 输出 0
        embeddings1_gpu = cp.asarray(embeddings1)
        embeddings2_gpu = cp.asarray(embeddings2)

        product = cp.dot(embeddings1_gpu, embeddings2_gpu.T)

        norm1 = cp.linalg.norm(embeddings1_gpu, axis=1, keepdims=True)
        norm2 = cp.linalg.norm(embeddings2_gpu, axis=1, keepdims=True)

        norm_product = cp.dot(norm1, norm2.T)

        cosine_similarities = product / norm_product

        return cp.asnumpy(cosine_similarities)

    def semantic_pruning_triplets(
        self, question, triplets, rel_embeddings=None, topk=30
    ):
        time_query = -time.time()
        question_embed = np.array(self.get_text_embedding(question)).reshape(1, -1)
        time_query += time.time()
        # print(f"query_embedding {time_query}")

        if rel_embeddings is None:
            time_triplet_embedding = -time.time()
            rel_embeddings = self.get_text_embeddings(triplets)
            time_triplet_embedding += time.time()
            print(f"kg_embedding cost {time_triplet_embedding:.3f}s")

        if len(rel_embeddings) == 1:
            rel_embeddings = np.array(rel_embeddings).reshape(1, -1)
        else:
            rel_embeddings = np.array(rel_embeddings)

        time_start_cp = -time.time()
        similarity_cp = self.cosine_similarity_cp(question_embed, rel_embeddings)[0]
        time_start_cp += time.time()
        similarity = similarity_cp

        time_sort_time = -time.time()
        all_rel_scores = [
            (rel, score) for rel, score in zip(triplets, similarity.tolist())
        ]
        sorted_all_rel_scores = sorted(all_rel_scores, key=lambda x: x[1], reverse=True)
        time_sort_time += time.time()
        # print_text(f"sorted cost {time_start}\n", color='red')

        return sorted_all_rel_scores[:topk]



class RetrieverEntities(object):
    def __init__(self,graphdb : GraphDatabase,entities_db : EntitiesDB):
        self.entities_db = entities_db
        self.graphdb = graphdb
        self.prunner = Pruning()
        
    def retrieve(
        self,
        question,
        limit=30,
        pruning=30,
        depth=2,
        entnum=10,
    ):
        q_embeddings = self.entities_db.get_embedding(question)
        ids, distances = self.entities_db.search(q_embeddings, limit=entnum)
        # print(f'question: {question}')
        # print(f'ids: {ids}')
        # print(f'distances: {distances}')
        similary_entities = [self.entities_db.id2entity[id] for id in ids]
        # print(f"similary_entities: {similary_entities}")
        entities = similary_entities

        print_text(f"question: {question}\n", color="red")
        print_text(f"entities: {entities}\n", color="red")

        all_rel_map = {}
        for entity in entities:
            rel_map = self.graphdb.get_rel_map(entities=[entity], depth=depth, limit=limit)
            # print('graph query', entity, '\n', rel_map, '\n')
            all_rel_map.update(rel_map)
        clean_rel_map = self.graphdb.clean_rel_map(all_rel_map)

        knowledge_sequences = []

        for k, v in clean_rel_map.items():
            # print(k, type(v))
            kg_seqs = self.graphdb.get_knowledge_sequence({k: v})
            knowledge_sequences.append(kg_seqs)

        # print_text(f"knowledge_sequences: {len(knowledge_sequences)}\n",
        #             color='red')

        # from utils.pruning import semantic_pruning_triplets

        print("knowledge_sequences",[len(x) for x in knowledge_sequences])

        if pruning > 0:
            knowledge_sequences_pruning = []
            for kg_seqs in knowledge_sequences:

                # sorted_all_rel_scores = semantic_pruning_triplets(question,
                sorted_all_rel_scores = self.prunner.semantic_pruning_triplets(
                    question, kg_seqs, rel_embeddings=None, topk=pruning
                )

                kg_seqs = [rel for rel, _ in sorted_all_rel_scores]
                knowledge_sequences_pruning.append(kg_seqs)

            print([len(x) for x in knowledge_sequences_pruning])

            knowledge_sequences = knowledge_sequences_pruning
            
        return knowledge_sequences



# keywords: ['2022', 'Tour de france', 'Won', 'Apple']
# question: Who won the 2022 Tour de France?
# if __name__ == "__main__":
    