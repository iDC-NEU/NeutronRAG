import random
import time

import cupy as cp
import numpy as np
from llama_index.core.utils import print_text
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from llmragenv.Cons_Retri.Embedding_Model import EmbeddingEnv, Ollama_EmbeddingEnv

# embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-small-en-v1.5",
#                                    embed_batch_size=10)
embed_model = None


class Pruning:

    def __init__(
        self,
        model="BAAI/bge-small-en-v1.5",
        device="cuda:0",
        batch_size=10,
        embed_model=None,
        step=100,
    ):
        self.step = step
        if embed_model is not None:
            self.embed_model = embed_model
        else:
            self.embed_model = Ollama_EmbeddingEnv()

    def get_text_embedding(self, text):
        embedding = self.embed_model.get_embedding(text)
        return embedding

    def get_text_embeddings(self, texts):
        all_embeddings = []
        n_text = len(texts)
        for start in range(0, n_text, self.step):
            input_texts = texts[start : min(start + self.step, n_text)]
            embeddings = self.embed_model.get_embeddings(input_texts)
            all_embeddings += embeddings
        return all_embeddings

    def cosine_similarity_cp(
        self,
        embeddings1,
        embeddings2,
    ) -> float:
        # with cp.cuda.Device(0):
        #     arr1 = cp.array([1, 2, 3])
        #     print(cp.cuda.runtime.getDevice())  # 输出 0
        embeddings1_gpu = cp.asarray(embeddings1)
        embeddings2_gpu = cp.asarray(embeddings2)

        product = cp.dot(embeddings1_gpu, embeddings2_gpu.T)

        norm1 = cp.linalg.norm(embeddings1_gpu, axis=1, keepdims=True)
        norm2 = cp.linalg.norm(embeddings2_gpu, axis=1, keepdims=True)

        norm_product = cp.dot(norm1, norm2.T)

        cosine_similarities = product / norm_product

        return cp.asnumpy(cosine_similarities)

    def semantic_pruning_triplets(
        self, question, triplets, rel_embeddings=None, topk=30
    ):
        time_query = -time.time()
        question_embed = np.array(self.get_text_embedding(question)).reshape(1, -1)
        time_query += time.time()
        # print(f"query_embedding {time_query}")

        if rel_embeddings is None:
            time_triplet_embedding = -time.time()
            rel_embeddings = self.get_text_embeddings(triplets)
            time_triplet_embedding += time.time()
            print(f"kg_embedding cost {time_triplet_embedding:.3f}s")

        if len(rel_embeddings) == 1:
            rel_embeddings = np.array(rel_embeddings).reshape(1, -1)
        else:
            rel_embeddings = np.array(rel_embeddings)

        time_start_cp = -time.time()
        similarity_cp = self.cosine_similarity_cp(question_embed, rel_embeddings)[0]
        time_start_cp += time.time()
        similarity = similarity_cp

        time_sort_time = -time.time()
        all_rel_scores = [
            (rel, score) for rel, score in zip(triplets, similarity.tolist())
        ]
        sorted_all_rel_scores = sorted(all_rel_scores, key=lambda x: x[1], reverse=True)
        time_sort_time += time.time()
        # print_text(f"sorted cost {time_start}\n", color='red')

        return sorted_all_rel_scores[:topk]


def calculate_tfidf_cosine_similarity(sentence1, sentence2):
    sentences = [sentence1, sentence2]
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(sentences)
    similarity = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])
    return similarity[0][0]


def calculate_embedding_cosine_similarity(sentence1, sentence2):
    global embed_model
    if not embed_model:
        embed_model = Ollama_EmbeddingEnv()
        print_text("pruning create embed_model\n", color="red")
    embed1 = np.array(embed_model.get_text_embedding(sentence1)).reshape(1, -1)
    embed2 = np.array(embed_model.get_text_embedding(sentence2)).reshape(1, -1)
    similarity = cosine_similarity(embed1, embed2)
    return similarity[0][0]


def get_text_embedding(text):
    global embed_model
    if not embed_model:
        embed_model = Ollama_EmbeddingEnv()
        print_text("create embed_model\n", color="red")
    # embedding = embed_model._get_text_embedding(text)
    embedding = embed_model.get_embedding(text)
    return embedding


def get_text_embeddings(texts, step=400, device=0):
    global embed_model
    if not embed_model:
        embed_model = Ollama_EmbeddingEnv()
        print_text("create embed_model\n", color="red")

    all_embeddings = []
    n_text = len(texts)
    # time_s = time.time()
    for start in range(0, n_text, step):
        input_texts = texts[start : min(start + step, n_text)]
        # print('process', len(input_texts))
        embeddings = embed_model.get_embeddings(input_texts)

        # print('start', start)
        all_embeddings += embeddings
    # time_e = time.time()
    # print(f'get_text_embeddings cost ({n_text}) {time_e - time_s:.3f}')
    # print(len(all_embeddings))
    return all_embeddings


def cosine_similarity_np(
    embeddings1,
    embeddings2,
) -> float:
    product = np.dot(embeddings1, embeddings2.T)

    norm1 = np.linalg.norm(embeddings1, axis=1, keepdims=True)
    norm2 = np.linalg.norm(embeddings2, axis=1, keepdims=True)

    norm_product = np.dot(norm1, norm2.T)
    cosine_similarities = product / norm_product

    return cosine_similarities


def cosine_similarity_cp(
    embeddings1,
    embeddings2,
) -> float:
    embeddings1_gpu = cp.asarray(embeddings1)
    embeddings2_gpu = cp.asarray(embeddings2)

    product = cp.dot(embeddings1_gpu, embeddings2_gpu.T)

    norm1 = cp.linalg.norm(embeddings1_gpu, axis=1, keepdims=True)
    norm2 = cp.linalg.norm(embeddings2_gpu, axis=1, keepdims=True)

    norm_product = cp.dot(norm1, norm2.T)

    cosine_similarities = product / norm_product

    return cp.asnumpy(cosine_similarities)


def semantic_pruning_triplets(
    question, triplets, rel_embeddings=None, topk=30, device=0
):
    time_query = -time.time()
    question_embed = np.array(get_text_embedding(question)).reshape(1, -1)
    time_query += time.time()
    # print(f"query_embedding {time_query}")

    if rel_embeddings is None:
        time_triplet_embedding = -time.time()
        rel_embeddings = get_text_embeddings(triplets, device)
        time_triplet_embedding += time.time()
        print(f"kg_embedding cost {time_triplet_embedding:.3f}s")

    if len(rel_embeddings) == 1:
        rel_embeddings = np.array(rel_embeddings).reshape(1, -1)
    else:
        rel_embeddings = np.array(rel_embeddings)

    time_start_cp = -time.time()
    similarity_cp = cosine_similarity_cp(question_embed, rel_embeddings)[0]
    time_start_cp += time.time()

    similarity = similarity_cp

    # print_text(f'similarity_cp {time_start_cp}\n', color='red')

    time_sort_time = -time.time()
    all_rel_scores = [(rel, score) for rel, score in zip(triplets, similarity.tolist())]
    sorted_all_rel_scores = sorted(all_rel_scores, key=lambda x: x[1], reverse=True)
    time_sort_time += time.time()
    # print_text(f"sorted cost {time_start}\n", color='red')

    return sorted_all_rel_scores[:topk]


# def semantic_pruning(question, knowledge_sequence, clean_rel_map, topk=30):
def semantic_pruning(question, knowledge_sequence, topk=30):

    
    time_query = -time.time()
    question_embed = np.array(get_text_embedding(question)).reshape(1, -1)
    time_query += time.time()
    print(f"query_embedding {time_query}")

    time_kg_embedding = -time.time()
    rel_embeddings = get_text_embeddings(knowledge_sequence)
    time_kg_embedding += time.time()
    print(f"kg_embedding {time_kg_embedding}")
    
    # 如果检索结果为空
    if not rel_embeddings:
        print("No valid embeddings for knowledge_sequence, returning empty list.")
        return []
    
    if len(rel_embeddings) == 1:
        rel_embeddings = np.array(rel_embeddings).reshape(1, -1)
    else:
        rel_embeddings = np.array(rel_embeddings)

    # time_start = -time.time()
    # similarity = cosine_similarity_np(question_embed, rel_embeddings)[0]
    # time_start += time.time()

    # print(similarity)
    # print()

    time_start_cp = -time.time()
    print("question_embed",question_embed)
    print("rel_embeddings",rel_embeddings)
    similarity_cp = cosine_similarity_cp(question_embed, rel_embeddings)[0]
    time_start_cp += time.time()

    similarity = similarity_cp

    print_text(f"similarity_cp {time_start_cp}\n", color="red")

    # assert np.allclose(similarity_cp, similarity)

    # print(type(similarity_cp), type(similarity))

    time_start = -time.time()
    all_rel_scores = [
        (rel, score) for rel, score in zip(knowledge_sequence, similarity.tolist())
    ]
    sorted_all_rel_scores = sorted(all_rel_scores, key=lambda x: x[1], reverse=True)
    time_start += time.time()

    print_text(f"sorted cost {time_start}\n", color="red")

    return sorted_all_rel_scores[:topk]


if __name__ == "__main__":
    texts = [f"{random.randint(1, 10000)}" for _ in range(50010)]

    # q_embedding = np.array(get_text_embedding(question))
    # embed1 = np.array(get_text_embedding('hello word'))

    # similarity1 = cosine_similarity(q_embedding.reshape(1, -1).reshape(1, -1), embed1.reshape(1, -1))
    # similarity2 = cosine_similarity(q_embedding.reshape(1, -1).reshape(1, -1), embed1.reshape(1, -1))

    # similarity3 = embed_model.similarity(q_embedding, embed1)
    # print(similarity1, similarity2, similarity3)

    # time_s = time.time()
    # for text in texts:
    #     embed_model.get_text_embedding(text)
    # time_e = time.time()
    # print(f'embed time cost {time_e - time_s:.3f}')

    # time_s = time.time()
    # embeddings = embed_model._get_text_embeddings(texts)
    # time_e = time.time()
    # print(len(embeddings[0]))
    # print(embeddings[0][:10])
    # print(f'embed time cost {time_e - time_s:.3f}')

    question = "hhhh"
    embeddings = get_text_embeddings([question] + texts)

    q_embed = np.array(np.array(embeddings[0])).reshape(1, -1)
    embeds = np.array(np.array(embeddings[1:]))

    time_s = time.time()
    similarity = cosine_similarity(q_embed, embeds)
    time_e = time.time()
    print(similarity.shape)
    print(similarity[0][:10])
    print(f"similarity time cost {time_e - time_s:.3f}")

    time_s = time.time()
    similarity1 = cosine_similarity_np(q_embed, embeds)
    time_e = time.time()
    print(similarity.shape)
    print(f"similarity time cost {time_e - time_s:.3f}")
    print(similarity1[0][:10])

    # embed_model.similarity()
