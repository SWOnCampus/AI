from embeddings.sentence_transform import sentence_embedding
from storage.faiss_storage import sentence_embedding_save, search_similar_sentences
from storage.elastic_search_storage import create_database, save_qna_data, CompanySize, get_similar_qna_data, get_saved_doc_data_by_id, get_all_data, Category
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


test_question = "매출 증대를 위한 고객 행동 분석을 AI의 도입을 통해 해결한 사례에는 어떤 것들이 있나요?"
test_answer = "11번가는 고객의 행동 데이터를 분석하여 구매 주기가 길어진 고객에게 AI 기반의 맞춤 쿠폰 제공. 고객 이탈률을 10% 감소시키고, VIP 고객의 재구매율 증가."

category = Category.PainPoints
company_size = CompanySize.MEDIUM

def create_data():
    save_qna_data(company_size=company_size, question=test_question, answer=test_answer, category= category)


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
    # create_data()
    # response = get_similar_qna_data(company_size=CompanySize.LARGE, question=test_question, data_size=3)
    get_all_data()
    # get_saved_doc_data_by_id(ES_INDEX_NAME, "ySAcLpMBhs7LHBg3Nqdm")



