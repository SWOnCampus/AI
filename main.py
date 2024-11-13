from embeddings.sentence_transform import sentence_embedding
from storage.faiss_storage import sentence_embedding_save, search_similar_sentences
from storage.elastic_search_storage import create_database, save_qna_data, CompanySize, get_saved_doc_data

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

    # 저장할 문장 임베딩 후 저장
    # embeddings = sentence_embedding(sentences)
    # sentence_embedding_save(embeddings)
    #
    # 입력한 문장 기반 FAISS 인덱스 탐색
    # search_similar_sentences(query_sentences, 4, sentences);

    # Elastic Search DB 저장 테스트

    # DB 생성 (이미 존재하는 경우 무시)
    create_database()
    test_question = " 대기업이 AI를 도입할 때 주로 어떤 분야에 활용하며, 도입 시 고려해야 할 사항은 무엇인가요?"
    test_answer = "대기업은 AI를 활용하여 운영 효율성 향상, 고객 경험 개선, 제품 및 서비스 혁신 등을 추구합니다. 예를 들어, 삼성SDS는 생성형 AI를 통해 의료 분야에서 환자 문서 요약과 승인 절차 개선을 실현하였습니다. 도입 시에는 데이터 보안, 윤리적 이슈, 기존 시스템과의 통합 등을 고려해야 합니다."

    save_qna_data(company_size=CompanySize.LARGE, question=test_question, answer= test_answer)
    # get_saved_doc_data("qna_embedding_large", "SiBoJ5MBkx7bZs-aNzoy")



