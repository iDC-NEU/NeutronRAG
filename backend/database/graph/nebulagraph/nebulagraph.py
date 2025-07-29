
from logger import Logger
from database.graph.graph_database import GraphDatabase
from nebula3.gclient.net import ConnectionPool
from nebula3.Config import Config
from database.graph.nebulagraph.FormatResp import print_resp
# from nebula3.common import *
import os
import json
import numpy as np
import time
from utils.file_util import file_exist


class NebulaClient:

    def __init__(self):
        config = Config()
        config.max_connection_pool_size = 10

        self.connection_pool = ConnectionPool()
        ok = self.connection_pool.init([('127.0.0.1', 9669)], config)
        assert ok

        self.session = self.connection_pool.get_session('root', 'nebula')

    def __del__(self):
        if self.connection_pool:
            self.connection_pool.close()
        if self.session:
            try:
                self.session.release()
            except Exception as e:
                print(f"Unexpected error: {e}")

    def create_space(self, space_name):
        self.session.execute(
            f'CREATE SPACE IF NOT EXISTS {space_name}(vid_type=FIXED_STRING(256), partition_num=1, replica_factor=1);'
        )
        time.sleep(10)
        self.session.execute(
            f'USE {space_name}; CREATE TAG IF NOT EXISTS entity(name string);')
        self.session.execute(
            f'USE {space_name}; CREATE EDGE IF NOT EXISTS relationship(relationship string);'
        )
        self.session.execute(
            f'USE {space_name}; CREATE TAG INDEX IF NOT EXISTS entity_index ON entity(name(256));'
        )
        time.sleep(10)

    def drop_space(self, space_name):
        if not isinstance(space_name, list):
            space_name = [space_name]
        for space in space_name:
            self.session.execute(f'drop space {space}')

    def info(self, space_name):
        result = self.session.execute(
            f'use {space_name}; submit job stats; show stats;')
        print(result)
        print_resp(result)

    def count_edges(self, space_name):
        result = self.session.execute(
            f'use {space_name}; MATCH (m)-[e]->(n) RETURN COUNT(*);')
        print_resp(result)

    def show_space(self):
        result = self.session.execute('SHOW SPACES;')
        print_resp(result)

    def show_edges(self, space_name, limits):
        result = self.session.execute(
            f'use {space_name}; MATCH ()-[e]->() RETURN e LIMIT {limits};')
        print_resp(result)

    def clear(self, space_name):
        query = f'CLEAR SPACE {space_name};'
        self.session.execute(query)

    def save_triplets(self, space_name, file_path=None):
        if not file_path:
            file_path = space_name + '_triplets.json'

        all_triples = self.get_triplets(space_name=space_name)

        with open(file_path, 'w', encoding='utf-8') as file:
            json.dump(all_triples, file, ensure_ascii=False, indent=2)
            print(f'save {len(all_triples)} triples to {file_path}.')

    def get_triplets(self, space_name):
        print("################################",space_name)
        result = self.session.execute(
            f'use {space_name}; MATCH (n1)-[e]->(n2) RETURN n1, e, n2;')

        all_triples = []

        if result.row_size() > 0:
            for row in result.rows():
                values = row.values

                head, relation, tail = '', '', ''

                for value in values:
                    if value.field == 9:  # 对应 Vertex
                        vertex = value.get_vVal()
                        if not head:
                            head = vertex.vid.get_sVal().decode('utf-8')
                        else:
                            tail = vertex.vid.get_sVal().decode('utf-8')

                    elif value.field == 10:  # 对应 Edge
                        edge = value.get_eVal()
                        relation = edge.props.get(
                            b'relationship').get_sVal().decode('utf-8')

                triplet = [head, relation, tail]
                all_triples.append(triplet)
        else:
            print(f'No triplets found {space_name}.')

        all_triples = set(tuple(triplet) for triplet in all_triples)
        all_triples = [list(triplet) for triplet in all_triples]

        return all_triples
    

    def get_retrieve_triplets_1hop(self, space_name, entities: list[str]):
        self.session.execute(f'USE {space_name}')

        # 将 entities 列表中的字符串拼接为查询条件
        entities_str = ", ".join([f'"{entity}"' for entity in entities])

        # 修改查询语句，根据传入的 entities 列表查询三元组
        query = f'''
        MATCH (n)-[e1]->(o)
        WHERE id(n) IN [{entities_str}] OR id(o) IN [{entities_str}]
        RETURN n, e1, o LIMIT 30;
        '''

        result = self.session.execute(query)

        # 检查查询是否成功
        if not result.is_succeeded():
            print(f"Query failed: {result.error_msg()}")
            return []
        else:
            # 打印查询结果
            print("Query succeeded. Results:")
            
            # 用于存储三元组的列表
            triplets = []

            for row in result.rows():
                # 提取源节点信息
                node_source = row.values[0].get_vVal()  # 获取第一个 Value 对象的 Vertex
                source_name = node_source.tags[0].props[b'name'].get_sVal().decode('utf-8')  # 解码 name 属性

                # 提取关系边信息
                relationship = row.values[1].get_eVal()  # 获取第二个 Value 对象的 Edge
                relationship_name = relationship.props[b'relationship'].get_sVal().decode('utf-8')  # 解码关系属性

                # 提取目标节点信息
                node_destination = row.values[2].get_vVal()  # 获取第三个 Value 对象的 Vertex
                destination_name = node_destination.tags[0].props[b'name'].get_sVal().decode('utf-8')  # 解码 name 属性

                # 构造一个三元组字典
                # triple = {
                #     "source": source_name,
                #     "relationship": relationship_name,
                #     "destination": destination_name
                # }

                triple = f"{source_name}-{relationship_name}->{destination_name}"

                # 将三元组添加到列表中
                triplets.append(triple)
            
            return triplets

    def get_retrieve_triplets_2hop(self, space_name, entities: list[str]):
        self.session.execute(f'USE {space_name}')

        # 将 entities 列表中的字符串拼接为查询条件
        entities_str = ", ".join([f'"{entity}"' for entity in entities])

        # 修改查询语句，根据传入的 entities 列表查询三元组
        query = f'''
        MATCH (n)-[e1]->(o1)-[e2]->(o2)
        WHERE id(n) IN [{entities_str}] OR id(o2) IN [{entities_str}]
        RETURN n, e1, o1, e2, o2 LIMIT 30;
        '''

        result = self.session.execute(query)

        # 用于存储三元组的列表
        triplets = []

        # 检查查询是否成功
        if not result.is_succeeded():
            print(f"Query failed: {result.error_msg()}")
            return triplets
        else:
            # 打印查询结果
            print("Query succeeded. Results:")

            
            return triplets


from llama_index.legacy.graph_stores.nebulagraph import NebulaGraphStore
from llama_index.core import StorageContext
from llama_index.core import load_index_from_storage
from llama_index.core import KnowledgeGraphIndex
from llama_index.core.retrievers import (
    KnowledgeGraphRAGRetriever, )
from typing import Any, Dict, List, Optional, Tuple
from llama_index.core.utils import print_text
from llama_index.core.schema import (
    # BaseNode,
    # MetadataMode,
    NodeWithScore,
    # QueryBundle,
    TextNode,
)
import re
from llmragenv.Cons_Retri.Embedding_Model import EmbeddingEnv


class NebulaDB(GraphDatabase):

    def __init__(self,
                 server_url="127.0.0.1:9669",
                 server_username = "root",
                 server_password = "nebula",
                 space_name = "rgb",
                #  log_file='./database/nebula.log',
                 create=False,
                 verbose=False):
        #  verbose=False, retriever=False, llm_env=None):
        # self.log_file = log_file
        # self.server_ip = server_ip
        # self.server_port = server_port

        os.environ["NEBULA_ADDRESS"] = server_url
        os.environ["NEBULA_USER"] = server_username
        os.environ["NEBULA_PASSWORD"] = server_password  # default is "nebula"

        self.space_name = space_name
        self.edge_types = ['relationship']
        self.rel_prop_names = ['relationship']
        self.tags = ['entity']
        self.client = NebulaClient()
        self.verbose = verbose
        self.store: NebulaGraphStore = None

        try:
            self.store, self.storage_context = self.init_nebula_store()
        except Exception:
            print(
                f'please use NebulaClient().create() to create space {self.space_name}!!!\n\n\n'
            )
        # self.graph_schema = self.store.get_schema(refresh=None)

        self.retriever = None
        self.entities = self.get_all_entities()

        # self.triplet2id, self.triplet_embeddings = self.generate_embedding()
        # self.entities = self.get_all_entities()

    def __del__(self):
        del self.client

    def init_nebula_store(self):
        nebula_store = NebulaGraphStore(
            space_name=self.space_name,
            edge_types=self.edge_types,
            rel_prop_names=self.rel_prop_names,
            tags=self.tags,
        )
        storage_context = StorageContext.from_defaults(
            graph_store=nebula_store)
        return nebula_store, storage_context

    def upsert_triplet(self, triplet: Tuple[str, str, str]):
        self.store.upsert_triplet(*triplet)

    def get_storage_context(self):
        return self.storage_context

    def get_space_name(self):
        return self.space_name

    def get_index(self):
        return load_index_from_storage(self.storage_context)

    def process_docs(self,
                     documents,
                     triplets_per_chunk=10,
                     include_embeddings=True,
                     data_dir='./storage_graph',
                     extract_fn=None,
                     cache=True):

        # TODO: use rebel to extract the kg elements.

        # filter documents
        # filter_documents = [doc for doc in documents if not is_file_processed(self.log_file, doc.id_)]
        # print(f'filter {len(documents) - len(filter_documents)} documents, last {len(filter_documents)} documents.')
        # documents = filter_documents

        # if len(documents) == 0:
        #     return

        # kg_index = KnowledgeGraphIndex.from_documents(
        #     documents,
        #     storage_context=self.storage_context,
        #     max_triplets_per_chunk=triplets_per_chunk,
        #     space_name=self.space_name,
        #     edge_types=self.edge_types,
        #     rel_prop_names=self.rel_prop_names,
        #     tags=self.tags,
        #     include_embeddings=True,
        #     show_progress=True,
        # )
        print(os.getcwd(), data_dir)
        index_loaded = False
        if cache:
            try:
                storage_context = StorageContext.from_defaults(
                    persist_dir=data_dir, graph_store=self.store)
                kg_index = load_index_from_storage(
                    storage_context=storage_context,
                    # service_context=service_context,
                    max_triplets_per_chunk=triplets_per_chunk,
                    space_name=self.space_name,
                    edge_types=self.edge_types,
                    rel_prop_names=self.rel_prop_names,
                    tags=self.tags,
                    verbose=True,
                    show_progress=True,
                )
                index_loaded = True
                print(f"graph index load from {data_dir}.")
                return kg_index
            except Exception:
                index_loaded = False

        if not index_loaded:
            kg_index = KnowledgeGraphIndex.from_documents(
                documents,
                storage_context=self.storage_context,
                kg_triplet_extract_fn=extract_fn,
                # service_context=service_context,
                max_triplets_per_chunk=triplets_per_chunk,
                space_name=self.space_name,
                edge_types=self.edge_types,
                rel_prop_names=self.rel_prop_names,
                tags=self.tags,
                include_embeddings=True,
                show_progress=True,
            )
        if cache:
            kg_index.storage_context.persist(persist_dir=data_dir)
            print(f"kg index store to {data_dir}.")
        # for doc in documents:
        #     append_log(self.log_file, doc.id_)
        return kg_index

    def get_rel_map(self, entities, depth=2, limit=30):
        rel_map: Optional[Dict] = self.store.get_rel_map(entities,
                                                         depth=depth,
                                                         limit=limit)
        return rel_map

    def set_retriever(self, llm_env, limit=30):
        self.retriever = KnowledgeGraphRAGRetriever(
            storage_context=self.storage_context,
            graph_traversal_depth=2,
            retriever_mode='keyword',
            verbose=False,
            entity_extract_template=llm_env.keyword_extract_prompt_template,
            synonym_expand_template=llm_env.synonym_expand_prompt_template,
            # clean_kg_sequences_fn=self.clean_kg_sequences,
            max_knowledge_sequence=limit)

    def get_entities(self, query_str: str) -> List[str]:
        """Get entities from query string."""
        return self.retriever._get_entities(query_str)

    def _get_knowledge_sequence(
            self,
            entities: List[str]) -> Tuple[List[str], Optional[Dict[Any, Any]]]:
        return self.retriever._get_knowledge_sequence(entities)

    def build_nodes(
            self,
            knowledge_sequence: List[str],
            rel_map: Optional[Dict[Any, Any]] = None) -> List[NodeWithScore]:
        if len(knowledge_sequence) == 0:
            print_text("> No knowledge sequence extracted from entities.\n",
                       color='red')
            return []

        _new_line_char = ", "
        context_string = (
            # f"The following are knowledge sequence in max depth"
            # f" {self._graph_traversal_depth} "
            # f"in the form of directed graph like:\n"
            # f"`subject -[predicate]->, object, <-[predicate_next_hop]-,"
            # f" object_next_hop ...`"
            # f" extracted based on key entities as subject:\n"
            # f"{_new_line_char.join(knowledge_sequence)}"
            f"{_new_line_char.join(knowledge_sequence)}")

        if self.verbose:
            print_text(f"Graph RAG context:\n{context_string}\n", color="blue")

        rel_node_info = {
            "kg_rel_map": rel_map,
            "kg_rel_text": knowledge_sequence,
        }
        metadata_keys = ["kg_rel_map", "kg_rel_text"]

        if self.graph_schema != "":
            rel_node_info["kg_schema"] = {"schema": self.graph_schema}
            metadata_keys.append("kg_schema")
        node = NodeWithScore(node=TextNode(
            text=context_string,
            score=1.0,
            metadata=rel_node_info,
            excluded_embed_metadata_keys=metadata_keys,
            excluded_llm_metadata_keys=metadata_keys,
        ))
        return [node]

    def get_knowledge_sequence(self, rel_map):
        knowledge_sequence = []
        if rel_map:
            knowledge_sequence.extend([
                str(rel_obj) for rel_objs in rel_map.values()
                for rel_obj in rel_objs
            ])
        else:
            print("> No knowledge sequence extracted from entities.")
            return []
        return knowledge_sequence

    def clean_sequence(self,
                       sequence,
                       name_pattern=r'(?<=\{name: )([^{}]+)(?=\})',
                       edge_pattern=r'(?<=\{relationship: )([^{}]+)(?=\})'):
        '''
        kg result: 'James{name: James} -[relationship:{relationship: Joined}]-> Michael jordan{name: Michael jordan}'

        clean the kg result above to James -Joined-> Michael jordan
        '''
        names = re.findall(name_pattern, sequence)
        edges = re.findall(edge_pattern, sequence)
        assert len(names) == sequence.count('{name:'), sequence
        assert len(edges) == sequence.count('{relationship:')
        for name in names:
            sequence = sequence.replace(f'{{name: {name}}}', '')
        for edge in edges:
            sequence = sequence.replace(
                f'[relationship:{{relationship: {edge}}}]', f'{edge}')
        return sequence

    def clean_kg_sequences(self, knowledge_sequence):
        exit(0)  # remove this function, any dependency?
        # clean_knowledge_sequence = [
        #     self.clean_sequence(seq) for seq in knowledge_sequence
        # ]
        # return clean_knowledge_sequence

    def clean_rel_map(self, rel_map):
        name_pattern = r'(?<=\{name: )([^{}]+)(?=\})'
        clean_rel_map = {}
        for entity, sequences in rel_map.items():
            name = re.findall(name_pattern, entity)[0]
            clean_ent = entity.replace(f'{{name: {name}}}', '')
            clean_seq = [self.clean_sequence(seq) for seq in sequences]
            clean_rel_map[clean_ent] = clean_seq
        return clean_rel_map

    def drop(self):
        self.client.drop_space(self.space_name)

    def info(self):
        self.client.info(self.space_name)

    def count_edges(self):
        self.client.count_edges(self.space_name)

    def show_edges(self, limits=10):
        self.client.show_edges(self.space_name, limits)

    def clear(self):
        self.client.clear(self.space_name)

    def show_space(self):
        return self.client.show_space()

    def get_triplets(self):
        return self.client.get_triplets(self.space_name)

    def save_triplets(self, file_path=None):
        self.client.save_triplets(self.space_name, file_path)

    def get_all_entities(self):
        all_triplets = self.get_triplets()

        left_entities = [triplet[0] for triplet in all_triplets]
        right_entities = [triplet[2] for triplet in all_triplets]
        entities = set(left_entities + right_entities)

        print(f'triplets: {len(all_triplets)}, entities: {len(entities)}')
        # print(list(entities)[:10])
        return entities

    def generate_embedding(self):
        file_path = f'/home/hdd/dataset/rag-data/{self.space_name}-triplet-embedding.npz'

        if file_exist(file_path):
            print(f"load embedding from {file_path}")
            loaded_data = np.load(file_path, allow_pickle=True)
            triplet2id = loaded_data['triplet2id'].item()
            triplet_embeddings = loaded_data['triplet_embeddings']

            return triplet2id, triplet_embeddings

        all_triplets = self.get_triplets()
        triplet2id = {}
        all_triplets_str = []
        for i, triplet in enumerate(all_triplets):
            triplet_str = ' '.join(triplet)
            triplet2id[triplet_str] = i
            all_triplets_str.append(triplet_str)

        embed_model = EmbeddingEnv(embed_name="BAAI/bge-small-en-v1.5",
                                   embed_batch_size=10)

        all_embeddings = []

        step = 400
        n_triplets = len(all_triplets_str)
        for start in range(0, n_triplets, step):
            input_texts = all_triplets_str[start:min(start + step, n_triplets)]
            # print(input_texts)
            embeddings = embed_model.get_embeddings(input_texts)
            all_embeddings += embeddings
            # break

        # for i, triplet in enumerate(all_triplets_str):
        #     print(i, triplet)
        #     embedding = embed_model.get_embedding(triplet)
        #     assert np.allclose(embedding, all_embeddings[i], atol=1e-4), i

        all_embeddings_np = np.array(all_embeddings, dtype=float)

        np.savez(file_path,
                 triplet2id=triplet2id,
                 triplet_embeddings=all_embeddings_np)

        print(
            f'triplet embeddings ({all_embeddings_np.shape}) saved to {file_path}'
        )
        return triplet2id, all_embeddings_np

    def load_triplets_embedding(self, file_path):

        self.client.save_triplets(self.space_name, file_path)

    def execute(self, query):
        result = self.store.execute(query)
        return result

    def two_hop_parse_triplets(self, query):
        # 定义正则表达式模式
        two_hop_pattern1 = re.compile(
            r'(.+) <-(?<! )(.+?)(?<! )- (.+) -(?<! )(.+?)(?<! )-> (.+)')
        two_hop_pattern2 = re.compile(
            r'(.+) <-(?<! )(.+?)(?<! )- (.+) <-(?<! )(.+?)(?<! )- (.+)')
        two_hop_pattern3 = re.compile(
            r'(.+) -(?<! )(.+?)(?<! )-> (.+) -(?<! )(.+?)(?<! )-> (.+)')
        two_hop_pattern4 = re.compile(
            r'(.+) -(?<! )(.+?)(?<! )-> (.+) <-(?<! )(.+?)(?<! )- (.+)')

        one_hop_pattern5 = re.compile(r'(.+) -(?<! )(.+?)(?<! )-> (.+)')
        one_hop_pattern6 = re.compile(r'(.+) <-(?<! )(.+?)(?<! )- (.+)')

        match = two_hop_pattern1.match(query)
        if match:
            entity1, relation1, entity2, relation2, entity3 = match.groups()
            return [(entity2, relation1, entity1),
                    (entity2, relation2, entity3)]

        match = two_hop_pattern2.match(query)
        if match:
            entity1, relation1, entity2, relation2, entity3 = match.groups()
            return [(entity2, relation1, entity1),
                    (entity3, relation2, entity2)]

        match = two_hop_pattern3.match(query)
        if match:
            entity1, relation1, entity2, relation2, entity3 = match.groups()
            return [(entity1, relation1, entity2),
                    (entity2, relation2, entity3)]

        match = two_hop_pattern4.match(query)
        if match:
            entity1, relation1, entity2, relation2, entity3 = match.groups()
            return [(entity1, relation1, entity2),
                    (entity3, relation2, entity2)]

        match = one_hop_pattern5.match(query)
        if match:
            entity1, relation1, entity2 = match.groups()
            return [(entity1, relation1, entity2)]

        match = one_hop_pattern6.match(query)
        if match:
            entity1, relation1, entity2 = match.groups()
            return [(entity2, relation1, entity1)]

        assert False, query

    def rel_map_to_triplets(self, clean_map):
        all_triplets = set()
        for rels in clean_map.values():
            triplets, _ = self.two_hop_parse_multi_triplets(rels)
            all_triplets.update(triplets)
        return all_triplets

    def kg_seqs_to_triplets(self, kg_seqs):
        all_triplets = []
        for rel in kg_seqs:
            for triplet in self.two_hop_parse_triplets(rel):
                all_triplets.append(triplet)
        all_triplets = set(all_triplets)

        return all_triplets

    def two_hop_parse_multi_triplets(self, queries):
        triplets = []
        rel_to_entities = {}
        for query in queries:
            query_triplets = self.two_hop_parse_triplets(query)
            triplets += query_triplets
            if query not in rel_to_entities:
                rel_to_entities[query] = set()
            for triplet in query_triplets:
                rel_to_entities[query].add(triplet[0])
                rel_to_entities[query].add(triplet[2])
        return triplets, rel_to_entities
    

if __name__ == '__main__':
    space_name = 'rgb'
    client = NebulaClient()
    client.show_space()
    client.info(space_name)
    db = NebulaDB(space_name = space_name)
    print("NEBULA_ADDRESS:", os.environ["NEBULA_ADDRESS"])
    print("NEBULA_USER:", os.environ["NEBULA_USER"])
    print("NEBULA_PASSWORD:", os.environ["NEBULA_PASSWORD"])
    rel_map = db.get_rel_map([' '], depth=1, limit=3)
    print("rel_map",rel_map)