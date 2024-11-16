from embeddings.sentence_transform import sentence_embedding
from storage.faiss_storage import sentence_embedding_save, search_similar_sentences
from storage.elastic_search_storage import create_database, save_qna_data, CompanySize, get_similar_qna_data, get_saved_doc_data_by_id, get_all_data, Category, Industry
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


test_question = "딥러닝 기반의 고객 행동 분석 시스템"
test_answer = "고객 프로필 데이터: 인구통계학적 정보(연령, 성별, 지역, 직업 등), 회원 가입 정보(가입일, 회원 등급, 선호 카테고리 등); 거래 데이터: 구매 이력(구매한 상품, 구매 날짜, 구매 금액, 결제 수단 등), 반품 및 교환 기록(반품/교환한 상품, 사유, 처리 결과 등); 웹사이트 및 앱 활동 데이터: 페이지 방문 기록(방문한 페이지, 체류 시간, 클릭한 링크 등), 검색 기록(검색어, 검색 시간, 검색 결과 클릭 여부 등), 장바구니 활동(장바구니에 추가한 상품, 제거한 상품, 보관 기간 등); 고객 서비스 상호작용 데이터: 고객 문의 기록(문의 내용, 문의 채널(전화, 이메일, 채팅 등), 처리 상태 등), 피드백 및 리뷰(제품 및 서비스에 대한 평가, 리뷰 내용, 평점 등); 소셜 미디어 데이터: 소셜 미디어 상호작용(좋아요, 공유, 댓글 등), 브랜드 언급(브랜드에 대한 언급 내용, 감정 분석 결과 등); 위치 데이터: 오프라인 매장 방문 기록(방문한 매장, 방문 시간, 구매 여부 등), 위치 기반 서비스 이용 기록(GPS를 통한 위치 추적 데이터 등); 기타 데이터: 설문 조사 응답(고객 만족도 조사 결과, 선호도 조사 응답 등), 로그인 및 인증 기록(로그인 시간, 로그인 방법, 실패 시도 등)."

category = Category.Data
company_size = CompanySize.MEDIUM
industry = Industry.Retail

def create_data():
    save_qna_data(company_size=company_size, question=test_question, answer=test_answer, category= category, industry= industry)


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



