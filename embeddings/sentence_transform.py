from sentence_transformers import SentenceTransformer

# KoSentenceBERT 모델 로드
# model = SentenceTransformer('snunlp/KR-SBERT-V40K-klueNLI-augSTS')

model = SentenceTransformer('distiluse-base-multilingual-cased-v1')

def sentence_embedding(sentences):
    # 문장 임베딩 생성
    embeddings = model.encode(sentences)
    return embeddings






