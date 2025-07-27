import asyncio
from datetime import datetime

import json
import random
from typing import AsyncGenerator, List, Tuple
import aiohttp
import time
import numpy as np
# /home/lipz/NeutronRAG/NeutronRAG/backend/llmragenv/Cons_Retri/Embedding_Model.py


from llmragenv.Cons_Retri.Embedding_Model import EmbeddingEnv



prompts = {
    "short": "What is the capital of France?",
    "medium": "Explain the theory of relativity in simple terms suitable for a 10-year-old child.",
    "long": "Summarize the causes and consequences of the French Revolution in detail, including social, political, and economic factors, and compare it with other revolutions of the same period.",
}


def get_rgb_question():
    file_path = "/home/lipz/NeutronRAG/NeutronRAG/data/single_hop/specific/single_entity/rgb/rgb.json"
    
    query_list = []

    with open(file_path, "r", encoding="utf-8") as f:
        data = json.load(f)
        for item in data:
            if "query" in item:
                query_list.append(item["query"])

    return query_list


def random_ip():
    return ".".join(str(random.randint(1, 255)) for _ in range(4))


async def get_request(
    input_requests: List[str],
    request_rate: float,
    cv: int,
) -> AsyncGenerator[str, None]:
    
    num_reqs = len(input_requests)
    input_requests = iter(input_requests)
    gamma_shape = (1/cv)**2
    scale = 1/(request_rate*gamma_shape)
    
    
    intervals = []
    print('create arrivals...')
    for i in range(num_reqs):
        interval = np.random.gamma(gamma_shape, scale)
        intervals.append(interval)
    intervals = np.array(intervals)
    count = 0
    
    for request in input_requests:
        yield request

        if request_rate == float("inf"):
            # If the request rate is infinity, then we don't need to wait.
            continue
    
        # Sample the request interval from the exponential distribution.
        interval = intervals[count]
        count += 1
        # The next request will be sent after the interval.
        await asyncio.sleep(interval)



async def send_embedding_request(query: str, fake_ip: str):
    url = "http://localhost:11434/v1/embeddings"
    headers = {
        "Content-Type": "application/json",
        "X-Client-IP": fake_ip
    }
    payload = {
        "model": "qllama/bge-large-en-v1.5:f16",  # 改成你在 ollama 中实际使用的模型名
        "input": query
    }

    async with aiohttp.ClientSession() as session:
        async with session.post(url, headers=headers, json=payload) as resp:
            if resp.status == 200:
                data = await resp.json()
                return data["data"][0]["embedding"]
            else:
                print(f"[{fake_ip}] Request failed with status {resp.status}")
                return None

async def test_with_ollama():
    query_list = get_rgb_question()
    print(f"共读取 {len(query_list)} 个 query")

    async for query in get_request(query_list, request_rate=5.0, cv=1):
        now = datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
        fake_ip = random_ip()
        print(f"[{now}] From {fake_ip} | Query: {query[:40]}...")

        embedding = await send_embedding_request(query, fake_ip)
        if embedding:
            print(f"→ Embedding length: {len(embedding)}")




if __name__ == "__main__":
    # 读取 query 列表
    # query_list = get_rgb_question()
    # print(f"共读取 {len(query_list)} 个 query")

    # # 异步测试 get_request 函数
    # async def test():
    #     async for query in get_request(query_list, request_rate=5.0, cv=1):
    #         now = datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
    #         print(f"[{now}] Query Arrived: {query}")

    # asyncio.run(test())

    asyncio.run(test_with_ollama())

