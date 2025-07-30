'''
Author: lpz 1565561624@qq.com
Date: 2025-07-29 20:36:03
LastEditors: lpz 1565561624@qq.com
LastEditTime: 2025-07-30 16:36:14
FilePath: /lipz/NeutronRAG/NeutronRAG/backend/experiment/banckend_experiment.py
Description: 这是默认设置,请设置`customMade`, 打开koroFileHeader查看配置 进行设置: https://github.com/OBKoro1/koro1FileHeader/wiki/%E9%85%8D%E7%BD%AE
'''
import asyncio
import json
import random
import time
from datetime import datetime
from collections import deque
from typing import List

from experiment.ollama_test import get_rgb_question
from schedule.schedular import Schedular 
from llmragenv.demo_chat import Demo_chat
from schedule.request import *


    

# 模拟异步请求生成器
async def simulate_incoming_requests(schedular: Schedular, query_list: List[str], user_prefix: str = "u", rate: float = 3.0):
    for i, query in enumerate(query_list):
        req = Request(
            user_id=f"{user_prefix}_{i}",
            query_id=f'{i}',
            query=query,
            dataset_name = 'rgb',
            dataset_path = '11',
            vector_retrieval=None,
            graph_retrieval=None,
            hybrid_retrieval = None,
            state="waiting",
            timestamp=time.time(),
            answer = 'yes',
            model_name = 'qwen-plus',
        )
        schedular.add_request(req)
        await asyncio.sleep(random.expovariate(rate))

# 异步处理检索的调度逻辑
async def scheduler_process(schedular: Schedular):
    while True:
        schedular.handle_batch(batch_size=4)
        await asyncio.sleep(0.5)

async def test_queue():
    schedular = Schedular()
    query_list = get_rgb_question()[:10]  # 只取前10个 query 做简单测试
    await simulate_incoming_requests(schedular, query_list, rate=2.0)



async def run_experiment():
    scheduler = Schedular()
    query_list = get_rgb_question()[:8]  # 只取前8个问题做测试

    tasks = [
        simulate_incoming_requests(scheduler, query_list, rate=3),
        scheduler_process(scheduler),
    ]
    await asyncio.gather(*tasks)

if __name__ == "__main__":
    # asyncio.run(test_queue())
    asyncio.run(run_experiment())