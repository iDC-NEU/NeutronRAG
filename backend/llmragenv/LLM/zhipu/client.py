'''
Author: lpz 1565561624@qq.com
Date: 2025-02-09 18:31:22
LastEditors: lpz 1565561624@qq.com
LastEditTime: 2025-08-06 16:00:43
FilePath: /lipz/NeutronRAG/NeutronRAG/backend/llmragenv/LLM/zhipu/client.py
Description: 这是默认设置,请设置`customMade`, 打开koroFileHeader查看配置 进行设置: https://github.com/OBKoro1/koro1FileHeader/wiki/%E9%85%8D%E7%BD%AE
'''
from typing import List, Dict, Optional
from zai import ZhipuAiClient
from overrides import override  # 如果你用到了 override 装饰器
from llmragenv.LLM.openai.client import OpenAIClient
from llmragenv.LLM.llm_base import LLMBase

class ZhipuClient(LLMBase):
    """
    Zhipu AI Client 封装类，对齐 OpenAIClient 风格，支持同步与流式响应
    """
    def __init__(self, model_name, url, key):
        super().__init__()
        self.client = ZhipuAiClient(
            api_key = key
        )
        self.model_name  = model_name

        print(f"✅ Use ZhipuAI backend to generate (model: {model_name})")

    def construct_messages(self, prompt: str, history: List[List[str]]) -> List[Dict[str, str]]:
        messages = []
        for user_input, ai_response in history:
            messages.append({"role": "user", "content": user_input})
            messages.append({"role": "assistant", "content": ai_response})
        messages.append({"role": "user", "content": prompt})
        return messages

   
    def chat_with_ai(self, prompt: str, history: Optional[List[List[str]]] = None) -> Optional[str]:
        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=self.construct_messages(prompt, history or []),
            temperature=0,
            max_tokens=1024
        )
        return response.choices[0].message.content

    
    def chat_with_ai_stream(self, prompt: str, history: Optional[List[List[str]]] = None):
        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=self.construct_messages(prompt, history or []),
            temperature=0.95,
            top_p=0.7,
            max_tokens=1024,
            stream=True,
            thinking={"type": "enabled"},
        )

        result = ""
        for chunk in response:
            content = chunk.choices[0].delta.content or ""
            result += content
            yield result
