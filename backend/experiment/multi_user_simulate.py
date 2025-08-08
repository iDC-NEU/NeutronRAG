import asyncio
import json
import random
import datetime
from multiprocessing import Process
import time
from llmragenv.demo_chat import Demo_chat

API_KEYS = [
    "21c044e890fb42c1a3a04e40d88b579a.ecyDXEVwZqoQCjgj",
    "6356c20c06434a13a60ebed7535f3bab.QZ3pzhU1llKhJlAn"
]

MODEL_NAME = "glm-4.5"
DATASET_NAME = "rgb"
DATASET_PATH = "/home/lipz/NeutronRAG/NeutronRAG/data/single_hop/specific/single_entity/rgb/rgb.json"
TABLE_SUFFIX_BASE = "test_session"

NUM_PROCESSES = 20  # 可根据机器调整

async def simulate_client(index):
    key = random.choice(API_KEYS)
    table_suffix = f"{TABLE_SUFFIX_BASE}_{index}"

    try:
        start_time = datetime.datetime.now()
        print(f"🟢 [{index}] Start at {start_time.strftime('%H:%M:%S.%f')}")

        current_model = Demo_chat(
            model_name=MODEL_NAME,
            api_key=key,
            dataset_name=DATASET_NAME,
            dataset_path=DATASET_PATH,
            path_name=table_suffix
        )

        
        for item in current_model.new_history_chat():
            try:
                print("qqq")
            except Exception as e:
                print(f"❌ JSON 打印错误：{e}")
                print(item)
        
        print("44")

    except Exception as e:
        error_time = datetime.datetime.now()
        print(f"❌ [{index}] Error at {error_time.strftime('%H:%M:%S.%f')} | Error: {e}")


def run_client(index):
    asyncio.run(simulate_client(index))


if __name__ == "__main__":
    start_time = time.time()  # 记录开始时间

    processes = []

    for i in range(NUM_PROCESSES):
        p = Process(target=run_client, args=(i,))
        p.start()
        processes.append(p)

    for p in processes:
        p.join()

    end_time = time.time()  # 记录结束时间
    elapsed_time = end_time - start_time

    print(f"\n🌐 多进程 + 异步并发测试完成，共 {NUM_PROCESSES} 个请求。")
    print(f"⏱️ 总耗时：{elapsed_time:.2f} 秒")