from sentence_transformers import SentenceTransformer
import faiss
# KoSentenceBERT 모델 로드
model = SentenceTransformer('snunlp/KR-SBERT-V40K-klueNLI-augSTS')

# KoSentenceBERT 모델의 문장 임베딩 벡터 길이
embedding_dimension = 768

# 코사인 유사도 기반 FAISS 인덱스 생성
index = faiss.IndexFlatIP(embedding_dimension)
def sentence_embedding(sentences):
    # 문장 임베딩 생성
    embeddings = model.encode(sentences)
    return embeddings


def sentence_embedding_save(sentences):
    embeddings = sentence_embedding(sentences)

    # 임베딩 된 문장 데이터를 FAISS에 저장
    index.add(embeddings)

def isInit():
    if(model is None):
        return False
    if(index is None):
        return False

    return True






