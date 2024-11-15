from embeddings.sentence_transform import sentence_embedding
from storage.faiss_storage import sentence_embedding_save, search_similar_sentences
from storage.elastic_search_storage import create_database, save_qna_data, CompanySize, get_similar_qna_data, get_saved_doc_data_by_id, get_all_data
from constants import ES_INDEX_NAME

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
    # create_database()
    test_question = "대규모 데이터를 처리하는 데 가장 적합한 딥러닝 플랫폼은 무엇인가요?"
    # test_answer = "대규모 데이터와 실시간 예측이 요구되는 환경에서는 구글의 Vertex AI나 아마존의 SageMaker와 같은 클라우드 기반 플랫폼이 적합합니다. Vertex AI는 통합 환경을 제공하여 모델 개발부터 배포까지의 전 과정을 한곳에서 관리할 수 있습니다. 대규모 연산을 처리하기 위해 다양한 GPU 및 TPU 옵션을 제공하며, 빅데이터와의 결합도 용이합니다. 또한 SageMaker는 분산 학습과 호환성이 뛰어나고, 실시간 예측 API를 제공하여 예측 결과를 바로 활용할 수 있습니다."

    # save_qna_data(company_size=CompanySize.LARGE, question=test_question, answer= test_answer)
    response = get_similar_qna_data(company_size=CompanySize.LARGE, question=test_question, data_size=3)
    # get_all_data()
    # get_saved_doc_data_by_id(ES_INDEX_NAME, "ySAcLpMBhs7LHBg3Nqdm")



