from bert_score import score
from embeddings.sentence_transform import sentence_embedding
from scipy.spatial.distance import cosine


HALLUCINATION_BOUND = 0.6

def validate_hallucination(rag_data, answer):

    print("::::: 입력 RAG 데이터 :::::\n")
    print(rag_data + "\n\n")

    print("::::: LLM 반환 내용 :::::\n")
    print(answer + "\n\n")

    # BERTScore 계산
    P, R, F1 = score([answer], [rag_data], lang="ko", verbose=False)

    # 결과 출력
    print(f"Precision: {P.mean().item():.4f}")
    print(f"Recall: {R.mean().item():.4f}")
    print(f"F1 Score: {F1.mean().item():.4f}")

    # print(f":::: 할루시네이션 검증 유사도 : {similarity * 100}% ::::\n")

    if(F1.mean().item() >= HALLUCINATION_BOUND):
        return True
    else:
        return False



def validate_hallucination_cosine_similarity(rag_data, answer):
    rag_embedding = sentence_embedding(rag_data)
    answer_embedding = sentence_embedding(answer)
    print("::::: 입력 RAG 데이터 :::::\n")
    print(rag_data + "\n\n")

    print("::::: LLM 반환 내용 :::::\n")
    print(answer + "\n\n")

    similarity = get_cosine_similarity(rag_embedding, answer_embedding)

    print(f":::: 할루시네이션 검증 유사도 : {similarity * 100}% ::::\n")

    if(similarity < HALLUCINATION_BOUND):
        return False
    return True


def get_cosine_similarity(vector1, vector2):
    cosine_similarity = cosine(vector1, vector2)
    return cosine_similarity
