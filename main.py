from src.sentence_transform import sentence_embedding_save, isInit
from src.search_sentences import search_similar_sentences
import time

# 임베딩할 문장 리스트 (벡터 DB)
sentences = [
    "AI를 활용한 비즈니스 혁신에 관심이 있습니다.",
    "회사의 데이터 분석 역량을 강화하고 싶습니다.",
    "스포츠 경기 결과를 알려주세요.",
    "주말에 좋은 여행지를 추천해 주세요."
]

# 입력으로 들어온 문장 리스트
query_sentences = [
    "데이터 기반 분석 기능을 추가하고 싶어요.",
    "많이 활용하는 AI를 사용하고 싶어요."
]


if __name__ == '__main__':
    # 초기화 대기
    while(not isInit()):
        time.sleep(1)

    # 저장할 문장 임베딩 후 저장
    sentence_embedding_save(sentences)

    # 입력한 문장 기반 FAISS 인덱스 탐색
    search_similar_sentences(query_sentences, 4, sentences);


