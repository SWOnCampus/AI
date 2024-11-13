from src.sentence_transform import sentence_embedding, index
import numpy as np


# 임베딩 벡터 정규화
def normalize_embeddings(embeddings):
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    return embeddings / norms
def search_similar_sentences(sentences, k, embedded_sentences):
    embeddings = sentence_embedding(sentences)
    # 입력 문장 임베딩 정규화
    normalized_embedding = normalize_embeddings(embeddings)

    # 유사한 벡터 검색
    distances, indices = index.search(normalized_embedding, k)

    print_result(sentences, distances, indices, embedded_sentences)

def print_result(sentences, distances, indices, embedded_sentences):
    # 검색 결과 출력 (코사인 유사도)
    print(f"입력 문장: {sentences}\n")
    for i, idx in enumerate(indices[0]):
        print(f"유사도 순위 {i+1}:")
        print(f"- 원본 문장: {embedded_sentences[idx]}")
        print(f"- 코사인 유사도: {distances[0][i]:.4f}")
        print("-" * 50)

