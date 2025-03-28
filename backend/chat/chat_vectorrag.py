'''
Author: fzb fzb0316@163.com
Date: 2024-09-20 13:37:09
LastEditors: lpz 1565561624@qq.com
LastEditTime: 2025-03-28 15:11:01
FilePath: /RAGWebUi_demo/chat/chat_vectorrag.py
Description: 这是默认设置,请设置`customMade`, 打开koroFileHeader查看配置 进行设置: https://github.com/OBKoro1/koro1FileHeader/wiki/%E9%85%8D%E7%BD%AE
'''



# from icecream import ic
from typing import List, Optional
from overrides import override


from chat.chat_base import ChatBase
from llmragenv.LLM.llm_base import LLMBase
from database.vector.vector_database import VectorDatabase
from llmragenv.Cons_Retri.Vector_Retriever import RetrieverVector

vectorrag_template = """
你是一个高度智能的系统，擅长根据用户的查询信息进行检索和生成精确的答案。你的任务是结合检索到的内容和你自身的知识，生成一个详细、连贯且有逻辑的中文回答。

用户的输入问题是：“{message}”，请根据检索到的信息和你的知识提供一个准确且详细的回复。

检索到的信息如下：“{nodes_text}”。请将这些信息与相关背景知识整合，生成一个丰富、准确的中文回答，并确保答案只包含中文。

"""

llama_prompt_template = (
    "You are an expert Q&A system that is trusted around the world. "
    "Always answer the query using the provided context information, and not prior knowledge. "
    "Some rules to follow:\n"
    "1. Never directly reference the given context in your answer.\n"
    "2. Avoid statements like 'Based on the context, ...' or 'The context information ...' or anything along those lines.\n"
    "Context information is below.\n"
    # "The following are knowledge sequence in max depth 2 in the form of `subject predicate, object, predicate_next_hop, object_next_hop ...` extracted based on key entities as subject:\n"

    # "The following are knowledge sequence in max depth 2 in the form of directed graph like:\n"
    # "`subject -[predicate]->, object, <-[predicate_next_hop]-, object_next_hop ...` extracted based on key entities as subject:\n"
    "---------------------\n"
    "{context}\n"
    "---------------------\n"
    # "Given the context information and not prior knowledge, answer the query. Note that only answer questions and explain what context information you used to answer, without any other explanation.\n"
    # "Given the context information and not prior knowledge, answer the query. Additionally, indicate how many pieces of contextual information were used in the answer and provide the specific content of that context in list form.\n"
    # "Given the context information and not prior knowledge, answer the query. Also, list the contextual information you used in your answer.\n"
    # "Given the context information and not prior knowledge, answer the query. Note that answer the query directly without explanation. Also, list the contextual information you used in your answer.\n"
    "Given the context information and not prior knowledge, answer the query. The answer to the question is a word or entity. If the provided information is insufficient to answer the question, respond 'Insufficient Information' Note that answer the query directly without explanation.\n"
    # "Also, list the contextual information you used in your answer.\n\n"
    # "Also, list the knowledge sequence you used in your answer, start with 'Knowledge sequence used:'.\n\n"
    "Query: {query}\n"
    "Answer:"
)


class ChatVectorRAG(ChatBase):

    def __init__(self, llm: LLMBase, vector_db: VectorDatabase):
        super().__init__(llm)
        self.vector_db = vector_db
        self.retriver_vector = RetrieverVector(llm,vector_db)

    # def retrieve_chunk():


    @override
    def retrieval_result(self):
        return self.retrieve_result
        

    @override
    def web_chat(self, message: str, history: List[Optional[List]] | None):
        
        self.retrieve_result = self.retriver_vector.retrieve(question=message)
        node_result = self.retriver_vector.format_chunks(self.retrieve_result)
        print("retrieve result:",node_result)

        prompt = llama_prompt_template.format(message = message,nodes_text = node_result)

        
        # ic(message)
        # ic(history)

        # answers = self._llm.chat_with_ai_stream(prompt, history)
        # result = ""
        # for chunk in answers:
        #     result =  result + chunk.choices[0].delta.content or ""

        #     yield result

        # ic(result)
        answers = self._llm.chat_with_ai(prompt)
        print("answers in vectorrag",answers)
        return answers


    @override
    def chat_without_stream(self, message: str):
        
        self.retrieve_result = self.retriver_vector.retrieve(question=message)
        node_result = self.retriver_vector.format_chunks(self.retrieve_result)
        # ic(node_result)
        prompt = llama_prompt_template.format(query = message, context = node_result)

        # ic(prompt)

        answers = self._llm.chat_with_ai(prompt)

        # ic(answers)

        return answers
