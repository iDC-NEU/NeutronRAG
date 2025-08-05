'''
Author: lpz 1565561624@qq.com
Date: 2025-08-03 08:09:32
LastEditors: lpz 1565561624@qq.com
LastEditTime: 2025-08-05 19:38:48
FilePath: /lipz/NeutronRAG/NeutronRAG/backend/database/vector/entitiesdb.py
Description: 这是默认设置,请设置`customMade`, 打开koroFileHeader查看配置 进行设置: https://github.com/OBKoro1/koro1FileHeader/wiki/%E9%85%8D%E7%BD%AE
'''
from tqdm import tqdm

from database.vector.Milvus.milvus import MilvusDB, myMilvus
from llmragenv.Cons_Retri.Embedding_Model import Ollama_EmbeddingEnv


class EntitiesDB:

    def __init__(
        self,
        entities,
        db_name = "rgb",
        embed_name="qllama/bge-large-en-v1.5:f16",
        overwrite=False,
        step=200,
        device="cuda:2",
        verbose=False,
    ):
        self.embed_model = Ollama_EmbeddingEnv(embed_name=embed_name, device=device)

        self.entities = sorted(list(entities))
        self.db_name = f"{db_name}_entities"

        self.id2entity = {i: entity for i, entity in enumerate(self.entities)}

        self.milvus_client = myMilvus()

        create_new_db = True

        if self.milvus_client.has_collection(self.db_name):
            print(f"exist {self.milvus_client.has_collection(self.db_name)}")
            print(f"count {self.milvus_client.get_vector_count(self.db_name)}")
            print(f"entities {len(entities)}")

        if (
            entities
            and self.milvus_client.has_collection(self.db_name)
            and self.milvus_client.get_vector_count(self.db_name) == len(entities)
        ):
            
            create_new_db = False
            print(f"{self.db_name} is existing!")

        # overwrite = overwrite or create_new_db
        overwrite = False

        if overwrite:
            assert (
                entities
            ), "need specify the entities when create new vector database."

        self.db = MilvusDB(
            db_name, 1024, overwrite=overwrite, metric="COSINE", verbose=False
        )
        if overwrite:
            # Strong, Bounded, Eventually, Session
            self.db.create(consistency_level="Strong")
            self.generate_embedding_and_insert(step=step)

        self.db.load()

    def generate_embedding_and_insert(self, step=150, start_num=0):
        print(f"start generate emebedding for {self.db_name} and insert to database...")
        # time.sleep(0.5)
        n_entities = len(self.entities)
        for i in tqdm(range(0, n_entities, step), f"insert vector to {self.db_name}"):
            start_idx = i
            end_idx = min(n_entities, i + step)
            print(start_idx, end_idx)
            print(start_idx, end_idx)
            embeddings = self.get_embedding(self.entities[start_idx:end_idx])
            ids = list(range(start_idx + start_num, end_idx + start_num))
            self.insert(ids, embeddings)
            assert len(ids) == len(embeddings)
            print(ids)
            if i % (step *  10) == 0:
                print(f'insert {len(ids)} vectors')

    def get_embedding(self, query):
        if isinstance(query, list):
            ret = self.embed_model.get_embeddings(query)
        else:
            ret = self.embed_model.get_embedding(query)
        return ret

    def search(self, query_embedding, limit=3):
        assert isinstance(query_embedding, list)
        if not isinstance(query_embedding[0], list):
            query_embedding = [query_embedding]
        ids, distances = self.db.search(query_embedding, limit=limit)
        return (ids, distances)

    def insert(self, id, query_embedding):
        if not isinstance(id, list):
            id = [id]
            query_embedding = [query_embedding]

        self.db.insert([id, query_embedding])

if __name__ == "__main__":
    db = EntitiesDB()