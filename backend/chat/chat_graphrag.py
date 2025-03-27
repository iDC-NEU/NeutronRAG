'''
Author: fzb fzb0316@163.com
Date: 2024-09-19 08:48:47
LastEditors: lpz 1565561624@qq.com
LastEditTime: 2025-03-27 11:02:30
FilePath: /RAGWebUi_demo/chat/chat_graphrag.py
Description: 这是默认设置,请设置`customMade`, 打开koroFileHeader查看配置 进行设置: https://github.com/OBKoro1/koro1FileHeader/wiki/%E9%85%8D%E7%BD%AE
'''



# from icecream import ic
from typing import List, Optional, Set
from overrides import override
from llama_index.core.prompts.base import PromptTemplate
from llama_index.core.prompts.prompt_type import PromptType


from chat.chat_base import ChatBase
from llmragenv.LLM.llm_base import LLMBase
from database.graph.graph_database import GraphDatabase
from llmragenv.Cons_Retri.KG_Retriever import RetrieverGraph



Graphrag_prompt_template = """
你是一个处理基于图数据的智能系统。下面是一条消息以及从知识图谱中提取的图三元组。请使用提供的三元组生成相关的回复或提取洞见。

消息：
{message}

图三元组：
{triplets}
Rules:

- Always response in Simplified Chinese, not English. or Grandma will be  very angry.

"""


llama_QA_system_prompt = (
    "You are an expert Q&A system that is trusted around the world. "
    "Always answer the query using the provided context information, and not prior knowledge. "
    "Some rules to follow:\n"
    "1. Never directly reference the given context in your answer.\n"
    "2. Avoid statements like 'Based on the context, ...' or 'The context information ...' or anything along those lines.\n"
    "Context information is below.\n"
    "---------------------\n")


llama_QA_graph_prompt = (
    f"{llama_QA_system_prompt} "
    "{context}\n"
    "---------------------\n"
    "Given the context information and not prior knowledge, answer the query. Note that only answer questions without explanation.\n"
    "Query: {query}\n"
    # "Given the context information answer the query.\n"
    "Answer:")

llama_QA_graph_prompt_with_one_context = (
    "You are an expert Q&A system that is trusted around the world. "
    "Always answer the query using the provided context information, and not prior knowledge. "
    "Some rules to follow:\n"
    "1. Never directly reference the given context in your answer.\n"
    "2. Avoid statements like 'Based on the context, ...' or 'The context information ...' or anything along those lines.\n"
    "Context information is below.\n"
    "{context}\n"
    "---------------------\n"
    "Given the context information and not prior knowledge, answer the query. Note that only answer questions without explanation.\n"
    "The context is only one, and if the contex does not contain the answer, you just need to output 'The context do not contain answer!'\n"
    "Query: {query}\n"
    "Answer:")


class ChatGraphRAG(ChatBase):

    def __init__(self, llm: LLMBase, graph_db : GraphDatabase):
        super().__init__(llm)
        self.graph_database = graph_db
        self.retriver_graph = RetrieverGraph(llm,graph_db)

    
    def retrieve_triplets(self, message, space_name):
        """
        Args:
            message (str): the query from users
            space_name (str): the graph name of graph database

        Returns:
            list[dict["source", "relationship", "destination"]]: retrieve triplets
        """
        self.triplets = self.retriver_graph.retrieve_2hop(question=message)
        return self.triplets
    
    def get_triplets(self):
        if self.triplets:
            return self.triplets
        else:
            return ["还未检索!"]
        
    @override
    def retrieval_result(self):
        return self.triplets

    @override
    def web_chat(self, message: str,history: List[Optional[List]] | None = None):
        
        self.triplets = self.retriver_graph.retrieve_2hop(question=message, pruning = 30)

        print("graph retrieval result",self.triplets)

        prompt = llama_QA_graph_prompt.format(query = message, context = self.triplets)

        # ic(prompt)
        # ic(history)

        # answers = self._llm.chat_with_ai_stream(prompt, history)
        # result = ""
        # for chunk in answers:
        #     result =  result + chunk.choices[0].delta.content or ""

        #     yield result

        # ic(result)
        return self._llm.chat_with_ai(prompt, history)

    def web_chat_with_triplets(self, message: str,triplets,history: List[Optional[List]] | None):

        prompt = llama_QA_graph_prompt.format(query = message, context = triplets)
        # ic(prompt)
        # ic(history)
        # result = 

        answers = self._llm.chat_with_ai_stream(prompt, history)
        result = ""
        for chunk in answers:
            result =  result + chunk.choices[0].delta.content or ""
            yield result

        # ic(result)
    
    @override
    def chat_without_stream(self, message: str, pruning = None):
        self.triplets = self.retriver_graph.retrieve_2hop(question=message, pruning = pruning)
        prompt = llama_QA_graph_prompt.format(query = message, context = self.triplets)
        # ic(prompt)

        answers = self._llm.chat_with_ai(prompt)
            
        # ic(answers)
        return answers
    

    def chat_without_stream_with_triplets(self, message: str, triplets = []):
        # ic(self.triplets)
        self.triplets = triplets

        prompt = llama_QA_graph_prompt.format(query = message, context = self.triplets)
        # ic(prompt)

        answers = self._llm.chat_with_ai(prompt)

        # ic(answers)
        return answers
    
    def chat_without_stream_with_one_triplet(self, message: str, triplet = ""):
        # ic(self.triplets)

        prompt = llama_QA_graph_prompt_with_one_context.format(query = message, context = triplet)
        # ic(prompt)

        answers = self._llm.chat_with_ai(prompt)

        # ic(answers)
        return answers

    def chat_without_rag(self, message: str):
        
        prompt = """
            You are a helpful, respectful, and honest assistant. Please help me answer the questions below.
            ---------------------
            {question}
            ---------------------
            """

        return self._llm.chat_with_ai(prompt.format(question=message))







if __name__ == "__main__":
    from database.graph.nebulagraph.nebulagraph import*
    from llmragenv.LLM.ollama.client import OllamaClient

    graph_db = NebulaDB()
    llm = OllamaClient(model_name="llama3:8b",url="http://localhost:11434/v1",key="ollama")

    chat_graph = ChatGraphRAG(llm,graph_db)
    print(chat_graph.retriver_graph.extract_keyword(question="Curry is a famous basketball player"))
    pruning_knowledge_sequence = chat_graph.retriver_graph.retrieve_2hop(question="Who won the 2022 Tour de France?")

    print(pruning_knowledge_sequence)



    
