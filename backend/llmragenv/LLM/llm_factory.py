'''
Author: fzb fzb0316@163.com
Date: 2024-09-20 13:37:09
LastEditors: lpz 1565561624@qq.com
LastEditTime: 2025-08-06 15:42:12
FilePath: /RAGWebUi_demo/llmragenv/LLM/llm_factory.py
Description: 这是默认设置,请设置`customMade`, 打开koroFileHeader查看配置 进行设置: https://github.com/OBKoro1/koro1FileHeader/wiki/%E9%85%8D%E7%BD%AE
'''

from llmragenv.LLM.baichuan.client import BaichuanClient
from llmragenv.LLM.client_error import ClientUrlFormatError, ClientAPIUnsupportedError, ClientError
from llmragenv.LLM.deepseek.client import DeepseekClient
from llmragenv.LLM.doubao.client import DoubaoClient
from llmragenv.LLM.lingyiwanwu.client import LingyiwanwuClient
from llmragenv.LLM.llm_base import LLMBase
from llmragenv.LLM.moonshot.client import MoonshotClient
from llmragenv.LLM.ollama.client import OllamaClient
from llmragenv.LLM.qwen.client import QwenClient
from llmragenv.LLM.zhipu.client import ZhipuClient
from llmragenv.LLM.openai.client import OpenAIClient
from utils.singleton import Singleton
from utils.url_paser import is_valid_url
from config.config import Config


LLMProvider = {
    "zhipu" : ["glm-4.5"],
    "baichuan" : [],
    "qwen" : ["qwen-plus","qwen-turbo-latest"],
    "moonshot" : [],
    "lingyiwanwu" : [],
    "deepseek" : ["deepseek-reasoner"],
    "doubao" : [],
    "gpt" : ["gpt-4o-mini"],
    "llama" : ["qwen:0.5b", "llama2:7b", "llama2:13b", "llama2:70b","qwen:7b","qwen:14b","qwen:72b","qwen:4b","llama3:8b"]
}

class ClientFactory(metaclass=Singleton):
    
    def __init__(self, model_name, url, key, llmbackend="openai"):
        """
        初始化 ClientFactory

        :param model_name: LLM 模型名称
        :param url: LLM API 访问 URL
        :param key: 访问 LLM API 所需的 Key
        :param llmbackend: LLM 后端类型（默认为 "openai"）
        """
        self.model_name = model_name
        self.url = url
        self.key = key
        self.llmbackend = llmbackend

    def get_client(self) -> LLMBase:
        """
        根据模型名称和 LLM 后端，返回相应的 LLM 客户端。

        :return: LLM 客户端实例
        :raises ClientAPIUnsupportedError: 如果模型名称不受支持
        :raises ClientError: 如果 llmbackend 不受支持
        """
        if self.llmbackend == "openai":
            if self.model_name in LLMProvider["zhipu"]:
                return ZhipuClient(self.model_name, self.url, self.key)

            elif self.model_name in LLMProvider["moonshot"]:
                return MoonshotClient(self.model_name, self.url, self.key)

            elif self.model_name in LLMProvider["baichuan"]:
                return BaichuanClient(self.model_name, self.url, self.key)

            elif self.model_name in LLMProvider["qwen"]:
                return QwenClient(self.model_name, self.url, self.key)

            elif self.model_name in LLMProvider["lingyiwanwu"]:
                return LingyiwanwuClient(self.model_name, self.url, self.key)

            elif self.model_name in LLMProvider["deepseek"]:
                print("deepseek-reasoner")
                return DeepseekClient(self.model_name, self.url, self.key)

            elif self.model_name in LLMProvider["doubao"]:
                return DoubaoClient(self.model_name, self.url, self.key)

            elif self.model_name in LLMProvider["gpt"]:
                return OpenAIClient(self.model_name, self.url, self.key)

            elif self.model_name in LLMProvider["llama"]:
                return OpenAIClient(self.model_name, self.url, self.key)

            else:
                raise ClientAPIUnsupportedError(f"No client API adapted for model: {self.model_name}")

        elif self.llmbackend == "llama_index":
            print("use llama-index")
            if self.model_name in LLMProvider["llama"]:
                return OllamaClient(self.model_name, self.url, self.key)

        else:
            raise ClientError(f"No llm_backend {self.llmbackend}")



# class ClientFactory(metaclass=Singleton):
    
#     def __init__(self, model_name, llmbackend = "openai"):
#         self.model_name = model_name
#         self.llmbackend = llmbackend


#     def get_client(self) -> LLMBase:
#         if self.llmbackend == "openai":
#             if self.model_name in LLMProvider["zhipu"]:
#                 url = Config.get_instance().get_with_nested_params("llm", "zhipu", "url")
#                 key = Config.get_instance().get_with_nested_params("llm", "zhipu", "key")
#                 return ZhipuClient()

#             elif self.model_name in LLMProvider["moonshot"]:
#                 url = Config.get_instance().get_with_nested_params("llm", "moonshot", "url")
#                 key = Config.get_instance().get_with_nested_params("llm", "moonshot", "key")
#                 return MoonshotClient()

#             elif self.model_name in LLMProvider["baichuan"]:
#                 url = Config.get_instance().get_with_nested_params("llm", "baichuan", "url")
#                 key = Config.get_instance().get_with_nested_params("llm", "baichuan", "key")
#                 return BaichuanClient()

#             elif self.model_name in LLMProvider["qwen"]:
#                 url = Config.get_instance().get_with_nested_params("llm", "qwen", "url")
#                 key = Config.get_instance().get_with_nested_params("llm", "qwen", "key")
#                 return QwenClient()

#             elif self.model_name in LLMProvider["lingyiwanwu"]:
#                 url = Config.get_instance().get_with_nested_params("llm", "lingyiwanwu", "url")
#                 key = Config.get_instance().get_with_nested_params("llm", "lingyiwanwu", "key")
#                 return LingyiwanwuClient()

#             elif self.model_name in LLMProvider["deepseek"]:
#                 url = Config.get_instance().get_with_nested_params("llm", "deepseek", "url")
#                 key = Config.get_instance().get_with_nested_params("llm", "deepseek", "key")
#                 return DeepseekClient()

#             elif self.model_name in LLMProvider["doubao"]:
#                 url = Config.get_instance().get_with_nested_params("llm", "doubao", "url")
#                 key = Config.get_instance().get_with_nested_params("llm", "doubao", "key")
#                 return DoubaoClient()

#             elif self.model_name in LLMProvider["gpt"]:
#                 url = Config.get_instance().get_with_nested_params("llm", "gpt", "url")
#                 key = Config.get_instance().get_with_nested_params("llm", "gpt", "key")
#                 return OpenAIClient(self.model_name, url, key)

#             elif self.model_name in LLMProvider["llama"]:
#                 url = Config.get_instance().get_with_nested_params("llm", "llama", "url")
#                 key = Config.get_instance().get_with_nested_params("llm", "llama", "key")
#                 return OpenAIClient(self.model_name, url, key)

#             else:
#                 raise ClientAPIUnsupportedError("No client API adapted")

#         elif self.llmbackend == "llama_index":
#             if self.model_name in LLMProvider["llama"]:
#                 url = Config.get_instance().get_with_nested_params("llm", "llama", "url")
#                 key = Config.get_instance().get_with_nested_params("llm", "llama", "key")
#                 return OllamaClient(self.model_name, url, key)
            
#         else:
#             raise ClientError(f"No llm_backend {self.llmbackend}")

if __name__ == "__main__":
    factory1 = ClientFactory()
    factory2 = ClientFactory()

    print(factory1 is factory2)
