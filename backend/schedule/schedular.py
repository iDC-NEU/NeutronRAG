'''
Author: lpz 1565561624@qq.com
Date: 2025-07-29 14:26:11
LastEditors: lpz 1565561624@qq.com
LastEditTime: 2025-07-30 16:09:04
FilePath: /lipz/NeutronRAG/NeutronRAG/backend/schedule/schedular.py
Description: 这是默认设置,请设置`customMade`, 打开koroFileHeader查看配置 进行设置: https://github.com/OBKoro1/koro1FileHeader/wiki/%E9%85%8D%E7%BD%AE
'''
import time
from collections import deque
from typing import Deque, List, Optional
from llmragenv.Cons_Retri.KG_Retriever import RetrieverGraph,RetrieverEntities
from concurrent.futures import ThreadPoolExecutor, as_completed
from database.vector.Milvus.milvus import MilvusDB, myMilvus
from llmragenv.demo_chat import Demo_chat
from schedule.request import *


WAITING = "waiting"
PROCESSING = "processing"
FINISHED = "finished"


class Schedular:
    def __init__(self):
        self.waiting_queue: Deque[Request] = deque()        # 等待处理的请求
        self.processing_queue: Deque[Request] = deque()     # 正在处理的请求
        self.finished: List[Request] = []                   # 完成的请求，准备返回给用户
    def add_request(self, request):
        """添加新请求"""
        self.waiting_queue.append(request)
        print(f"[{time.strftime('%H:%M:%S')}] New request queued: {request.user_id} → {request.query}")


    def dispatch_to_processing(self):
        """从 waiting_queue 分发到 processing_queue"""
        if self.waiting_queue:
            req = self.waiting_queue.popleft()
            self.processing_queue.append(req)
            req.state = PROCESSING
            print(f"Dispatching to retrieval: {req.user_id} → {req.query}")
            return req
        return None
    
    def finish_retrieval(self, req:Request):
        """检索处理完成，移动到 llm_queue"""
        if req in self.processing_queue:
            self.processing_queue.remove(req)
        req.state = "retrieved"
        self.finished.append(req)
        print(f"→ Retrieval complete for: {req.user_id} → {req.query}")

    def dispatch_to_llm(self):
        """从 llm_queue 中取出准备交给 LLM"""
        if self.finished:
            req = self.finished.popleft()
            req.state = "generating"
            print(f"Sending to LLM: {req.user_id} → {req.query}")
            return req
        return None
    
    def finish_request(self, req, llm_output=None):
        """LLM 生成完成，记录结果，标记为完成"""
        req.state = "finished"
        req.response = llm_output
        self.finished.append(req)
        print(f"✅ Request finished: {req.user_id} → {req.query}")


    def handle_batch(self,batch_size=4):
        if not self.waiting_queue:
            print("⚠️ No request in waiting_queue.")
            return
        
        batch = []
        for _ in range(min(batch_size, len(self.waiting_queue))):
            req = self.waiting_queue.popleft()
            req.state = PendingDeprecationWarning
            self.processing_queue.append(req)
            batch.append(req)


        def process(req):
            try:
                # 基于 Request 构造 Demo_chat 实例
                current_model = Demo_chat.from_request(req)
                # 处理请求（完成向量+图检索）
                current_model.handle_one_request(req=req)
                self.finished.append(req)
                return req, True
            except Exception as e:
                print(f"❌ Retrieval failed for {req.user_id}: {e}")
                return req, False

        with ThreadPoolExecutor(max_workers=batch_size) as executor:
            futures = [executor.submit(process, req) for req in batch]

            for future in as_completed(futures):
                req, success = future.result()
                self.processing_queue.remove(req)
                if success:
                    self.finished.append(req)
                    print(f"✅ Retrieval finished: {req.user_id} → {req.query}")
                else:
                    req.state = "error"
                    print(f"🚫 Marked as error: {req.user_id}")