import time
from typing import (
    # Optional, Dict,
    List)
# export PYTHONPATH=/home/lipz/fzb_rag_demo/RAGWebUi_demo/:$PYTHONPATH
from pymilvus import (
    connections,
    # utility,
    # FieldSchema,
    # CollectionSchema,
    DataType,
    Collection,
    Milvus,
    # MilvusClient,
    # Connections,
)
from database.vector.vector_database import VectorDatabase

from llama_index.vector_stores.milvus import MilvusVectorStore
import time
from typing import (
    # Optional, Dict,
    List)
from llama_index.core.utils import print_text

from llama_index.core import (
    VectorStoreIndex,
    # SimpleDirectoryReader,
    # Document,
    StorageContext,
    load_index_from_storage,
)

from pymilvus import (
    connections,
    # utility,
    # FieldSchema,
    # CollectionSchema,
    DataType,
    Collection,
    Milvus,
    # MilvusClient,
    # Connections,
)
from llama_index.core.schema import NodeWithScore, QueryBundle
from llama_index.core.retrievers import (
    VectorIndexRetriever)
from llama_index.vector_stores.milvus import MilvusVectorStore
import time
from typing import (
    # Optional, Dict,
    List)
from llama_index.core.utils import print_text
from llama_index.core import (
    VectorStoreIndex,
    # SimpleDirectoryReader,
    # Document,
    StorageContext,
    load_index_from_storage,
)

from pymilvus import (
    connections,
    # utility,
    # FieldSchema,
    # CollectionSchema,
    DataType,
    Collection,
    Milvus,
    # MilvusClient,
    # Connections,
)
from llama_index.core.schema import NodeWithScore, QueryBundle

from llama_index.core.retrievers import (

    # KnowledgeGraphRAGRetriever,
    VectorIndexRetriever)
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
fmt = "\n=== {:30} ===\n"



class MilvusClientTool:

    def __init__(
        self,
        server_ip='202.199.13.67',
        server_port='19530',
    ):
        self.client = Milvus(server_ip, server_port)

    def show_all_collections(self):
        print(fmt.format('all collections name'))
        ret = self.client.list_collections()
        print(ret)

    def show_collections_stats(self):
        print(fmt.format(f'stat of {self.collection_name}'))
        ret = self.client.get_collection_stats(self.collection_name)
        print(ret)

    def show_collections_schema(self):
        print(fmt.format(f'schema of {self.collection_name}'))
        ret = self.client.describe_collection(self.collection_name)
        print(ret)

    def clear(self, collection_name):
        print(fmt.format(f'clear collection {collection_name}'))
        self.client.drop_collection(collection_name)

class MilvusDB(VectorDatabase):

    def __init__(self,
                 collection_name,
                 dim,
                 overwrite=False,
                 similarity_top_k=5,
                 server_ip='127.0.0.1',
                 server_port='19530',
                 log_file='./database/milvus.log',
                 store=False,
                 verbose=False,
                 metric='COSINE',
                 retriever=False):
        self.collection_name = collection_name
        self.dim = dim
        self.overwrite = overwrite
        self.server_ip = server_ip
        self.server_port = server_port
        self.log_file = log_file

        self.client = Milvus(server_ip, server_port)
        # self.client = MilvusClient()

        self.store = None
        self.storage_context = None
        self.db = None
        self.verbose = verbose
        self.metric = metric

        self.index = None
        self.retriever = None
        self.topk = similarity_top_k

        self.embed_model = HuggingFaceEmbedding(
                model_name="BAAI/bge-large-en-v1.5",
                embed_batch_size=20,
                device="cuda:2")


        connections.connect("default", host="localhost", port=server_port)

        if store:
            self.init_store()

        if retriever:
            index = self.get_vector_index()
            self.retriever = VectorIndexRetriever(
                index=index, similarity_top_k=similarity_top_k)
            

    def get_storage_context(self):
        return self.storage_context

    def init_store(self):
        self.store = MilvusVectorStore(dim=self.dim,
                                       collection_name=self.collection_name,
                                       overwrite=self.overwrite)
        self.storage_context = StorageContext.from_defaults(
            vector_store=self.store)

    def get_vector_index(self):
        if not self.index:
            self.index = VectorStoreIndex.from_vector_store(
                vector_store=self.store,embed_model=self.embed_model)
        return self.index
    def clear(self):
        print(fmt.format(f'clear collection {self.collection_name}'))
        ret = self.client.drop_collection(self.collection_name)
        return ret

    def create(self, consistency_level="Session"):

        # connections.connect("default", host="localhost", port="19530")

        if self.overwrite and self.collection_name in self.client.list_collections(
        ):
            self.client.drop_collection(self.collection_name)

        # fields = [
        #     FieldSchema(name="pk", dtype=DataType.INT64, is_primary=True, auto_id=False),
        #     # FieldSchema(name="random", dtype=DataType.DOUBLE),
        #     FieldSchema(name="vec", dtype=DataType.FLOAT_VECTOR, dim=self.dim)
        # ]

        fields = {
            "fields": [{
                "name": "pk",
                "type": DataType.INT64,
                "is_primary": True
            }, {
                "name": "vec",
                "type": DataType.FLOAT_VECTOR,
                "params": {
                    "dim": self.dim
                }
            }],
            "auto_id":
            False
        }

        self.client.create_collection(
            self.collection_name,
            fields,
            consistency_level=consistency_level
            #   "Strong"
            #   Bounded
            #   Eventually
            #   "Session"
        )

        index = {
            "index_type": "IVF_FLAT",
            "metric_type": self.metric,
            "params": {
                "nlist": 128
            },
        }
        # self.db.create_index("embeddings", index)

        # schema = CollectionSchema(fields, "customize schema")

        # self.db = Collection(self.collection_name, schema)

        self.client.create_index(self.collection_name, 'vec', index)

        # print(Connections().list_connections)

        # print('has', self.client.has_collection(self.collection_name))

        # self.db = Collection(self.collection_name, consistency_level=consistency_level)
        self.db = Collection(self.collection_name)
        self.db.load()

    def show_collections_stats(self):
        print(fmt.format(f'stat of {self.collection_name}'))
        ret = self.client.get_collection_stats(self.collection_name)
        # ret = self.client.num_entities(self.collection_name)
        print(ret)

    def get_topk_vector(self, query_vector):
    # 加载集合
        collection = Collection(self.collection_name)

        # 定义搜索参数，根据度量类型调整
        search_params = {
            "metric_type": self.metric,
            "params": {
                "nprobe": 10
            },
        }

        # 检索与输入向量最相似的 top-k 向量
        results = collection.search(
            data=[query_vector],  # 输入查询向量
            anns_field="embedding",  # 向量字段名称
            param=search_params,
            limit=self.topk,  # 返回前 k 个相似向量
            expr=None  # 可选的筛选表达式
        )

        # 处理并返回检索结果
        topk_results = []
        for result in results:
            for hit in result:
                print(hit)
                topk_results.append({
                    "id": hit.id,  # 相似向量的 ID
                    "distance": hit.distance  # 与查询向量的距离
                })

        return topk_results
    
    def set_retriever(self, retriever):
        self.retriever = retriever

    def retrieve_nodes(self, query, embedding) -> List[NodeWithScore]:
        assert self.retriever, 'please use set_retriever() to init retriever!'

        query_bundle = QueryBundle(query_str=query, embedding=embedding)
        nodes = self.retriever._retrieve(query_bundle=query_bundle)
        return nodes





    
def test_retrieve_nodes(db_name):
    from llmragenv.Cons_Retri.Embedding_Model import  EmbeddingEnv

    vector_db = MilvusDB(db_name, 1024, overwrite=False, store=True,retriever=True)
    vector_db.show_collections_stats()
    collection = Collection("rgb")
    collection.load()
    # print(collection.is_loaded)

    question = "Tell me about Lebron James?"

    embed_model = EmbeddingEnv(embed_name="BAAI/bge-large-en-v1.5")
    embedding = embed_model.get_embedding(question)

    # vector_index = vector_db.get_vector_index()
    # vector_retriever = VectorIndexRetriever(index=vector_index)
    # vector_db.set_retriever(vector_retriever)

    nodes = vector_db.retrieve_nodes(question, embedding)

    # print(f'nodes:\n{nodes}')
    # for node in nodes:
    #     print_text(f'{node.text}\n', color='yellow')
    


if __name__ == "__main__":
    
    # import time
    # start = time.perf_counter()
    # vector_db = MilvusDB(
    #         collection_name="rgb",
    #         dim=1024,
    #         server_ip='127.0.0.1',
    #         server_port='19530',
    #         similarity_top_k=5  # 设置 top_k 为 5
    #     )

    # query_vector = [0.1] * 1024  

    # # 调用 get_topk_chunk 方法进行检索
    # topk_results = vector_db.get_topk_vector(query_vector)
    # end = time.perf_counter()
    # print(f"执行时间: {end - start:.4f} 秒")

    # # 打印结果
    # print("Top 5 similar vectors:")
    # for result in topk_results:
    #     print(f"ID: {result['id']}, Distance: {result['distance']}")
    #     print(result)

    # import time

    # db_name = "rgb"

    # start = time.perf_counter()
    # test_retrieve_nodes(db_name)
    # end = time.perf_counter()

    # print(f"执行时间: {end - start:.4f} 秒")
    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM
    import time

    model_path = "/home/hdd/model/Llama-2-13b-chat-hf"
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(
    model_path
    ).to("cuda")

    input_text = "介绍一下人工智能的历史和发展。"  # 示例输入

    # Tokenize 输入
    input_ids = tokenizer.encode(input_text, return_tensors="pt").to("cuda")
    start_time = time.time()

    # 生成 2500 token（max_new_tokens=2500）
    output = model.generate(
        input_ids,
        max_new_tokens=2500,  # 控制生成 token 数量
        do_sample=True,       # 启用随机采样
        temperature=0.7,      # 控制随机性
        top_p=0.9,            # Nucleus sampling
    )

    end_time = time.time()