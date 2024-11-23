from fastapi import FastAPI, Query, WebSocket, WebSocketDisconnect
from pydantic import BaseModel, Field
from embeddings.sentence_transform import sentence_embedding
from storage.faiss_storage import sentence_embedding_save, search_similar_sentences
from storage.elastic_search_storage import create_database, save_qna_data, CompanySize, get_similar_qna_data, get_saved_doc_data_by_id, get_all_data, Category, Industry
from llm.generate_result import create_consulting_result, get_summary_result
from constants import ES_INDEX_NAME
from globals import connected_clients
from log.send_log_data import send_info_test
import asyncio

# # 임베딩할 문장 리스트 (벡터 DB)
# sentences = [
#     "AI를 활용한 비즈니스 혁신에 관심이 있습니다.",
#     "회사의 데이터 분석 역량을 강화하고 싶습니다.",
#     "스포츠 경기 결과를 알려주세요.",
#     "주말에 좋은 여행지를 추천해 주세요."
# ]
#
# # 입력으로 들어온 문장 리스트
# query_sentences = [
#     "데이터 기반 분석 기능을 추가하고 싶어요.",
#     "많이 활용하는 AI를 사용하고 싶어요."
# ]
# test_question = "딥러닝 기반의 고객 행동 분석 시스템"
# test_answer = """
# 중견 소매 및 이커머스 기업이 딥러닝 기반의 고객 행동 분석 시스템을 도입하려면, 여러 요소를 고려하여 예산을 책정해야 합니다. 주요 비용 요소는 다음과 같습니다:
#
# 데이터 수집 및 전처리: 고객의 거래 기록, 웹사이트 클릭 패턴, 소셜 미디어 활동 등 다양한 데이터를 수집하고 정제하는 과정이 필요합니다. 이 단계에서는 데이터 라벨링 여부에 따라 비용이 달라질 수 있습니다. 라벨링이 되어 있지 않은 경우 추가 작업이 필요하여 비용이 증가할 수 있습니다.
# 숨고
#
# 모델 개발 및 학습: 고객 데이터를 분석하는 데 적합한 딥러닝 모델을 선택하고 학습시키는 과정입니다. 이 단계에서는 머신러닝 전문가의 인건비와 컴퓨팅 자원 비용이 포함됩니다. 숨고에 따르면, 인공지능(AI) 개발의 평균 비용은 약 100만 원이며, 최저 60만 원에서 최고 200만 원으로 책정됩니다.
# 숨고
#
# 시스템 통합 및 배포: 개발된 모델을 기존 시스템에 통합하고 실제 운영 환경에 배포하는 과정입니다. 이 단계에서는 시스템 통합 비용과 인프라 구축 비용이 발생합니다.
#
# 유지보수 및 업데이트: 시스템 운영 중 발생하는 문제를 해결하고, 새로운 데이터에 맞춰 모델을 업데이트하는 데 필요한 비용입니다.
#
# 이러한 요소들을 종합적으로 고려할 때, 중견기업이 딥러닝 기반의 고객 행동 분석 시스템을 도입하는 데 드는 총 예산은 수천만 원에서 수억 원에 이를 수 있습니다. 정확한 비용은 기업의 요구사항, 데이터의 양과 복잡성, 시스템의 규모 등에 따라 달라지므로, 전문 컨설팅 업체와 협의하여 상세한 견적을 받는 것이 좋습니다.
# """
# category = Category.CostROI
# company_size = CompanySize.MEDIUM
# industry = Industry.Retail
#

# API 막힘 임시 데이터
test_result = "1. 경쟁사 AI 도입 사례\n**카맥스**\n- **AI 도입 목적:**\n  - 고객 리뷰 요약 생성을 통해 고객이 빠르게 결정을 내릴 수 있도록 지원\n- **구체적인 성과:**\n  - 수천 개의 차량 리뷰를 몇 개월 안에 요약하여 고객 경험을 향상\n- **영향:**\n  - 고객들이 더 신속하고 효율적으로 구매 결정을 내릴 수 있게 도움\n\n**스티치 픽스**\n- **AI 도입 목적:**\n  - 고객의 스타일 선호도 수집 및 AI 기술 결합을 통한 맞춤화된 의류 추천\n- **구체적인 성과:**\n  - 패션 트렌드 분석, 효율적인 재고 관리 및 로지스틱스 최적화\n- **영향:**\n  - 고객들에게 개인화된 쇼핑 경험과 효율적인 재고 관리 제공\n\n**브이캣**\n- **AI 도입 목적:**\n  - 쇼핑몰 URL을 입력하면 자동 광고 소재 생성을 통한 광고 효율 증대\n- **구체적인 성과:**\n  - 300% 증가한 ROAS, 일부 광고 소재는 1,000% 이상의 성과\n- **영향:**\n  - 빠르고 다양한 광고 소재 제작을 통한 효율적인 마케팅 전략 구축\n\n**레코픽**\n- **AI 도입 목적:**\n  - 고객의 행동 데이터 분석을 통한 개인화된 상품 추천 제공\n- **구체적인 성과:**\n  - 매출 15.6% 증가, 구매 건수 16.7% 상승\n- **영향:**\n  - 고객의 구매 전환율 증가 및 매출 향상에 큰 도움을 줌\n\n2. Pain Point 관련 추천 AI\nAI 서비스: 이마트와 The Home Depot의 AI 기반 고객 행동 분석 시스템\n\n서비스에 대한 설명: \n이마트와 The Home Depot와 같이 소매업체들이 AI를 활용한 고객 행동 분석 시스템을 도입하였습니다. 이 시스템은 고객의 행동 데이터를 수집하고 분석하여 구매 패턴, 선호도, 관심사 등을 파악하여 맞춤 서비스 및 상품을 제공합니다. 이를 위해 딥러닝과 자연어 처리 기술을 활용하여 고객의 행동을 예측하고 개인화된 마케팅 전략을 수립합니다. 또한, 이를 통해 매장 내 고객 동선을 파악하고 상품 배치 및 프로모션 전략을 최적화하는데 활용됩니다.\n\n기대 효과: \n이러한 AI 기반 고객 행동 분석 시스템을 도입함으로써, 기업은 고객의 니즈를 더 잘 파악하고 맞춤형 서비스를 제공할 수 있습니다. 이를 통해 고객 만족도를 높이고 매출을 증대시킬 수 있습니다. 또한, 개인화된 추천 시스템을 통해 고객의 재방문율을 높이고 구매 전환율을 증가시킬 수 있습니다. 더불어, 고객 이탈률을 감소시키고 VIP 고객의 재구매율을 증가시킴으로써 고객 유지율을 높일 수 있습니다. 이를 통해 고객 참여도를 높이고 매출 성장을 이룰 수 있습니다. 따라서, AI 기반 고객 행동 분석 시스템은 소매업체의 매출 증대를 위한 효과적인 솔루션으로써 추천됩니다.\n\n3. AI 도입 시 필요 데이터\n**AI 기반 고객 행동 분석 시스템을 위한 데이터 목록**\n\n—\n\n**1. 고객 프로필 데이터**\n- 인구통계학적 정보\n  - 연령\n  - 성별\n  - 지역\n  - 직업\n- 회원 가입 정보\n  - 가입일\n  - 회원 등급\n  - 선호 카테고리\n\n**2. 거래 데이터**\n- 구매 이력\n  - 구매한 상품\n  - 구매 날짜\n  - 구매 금액\n  - 결제 수단\n- 반품 및 교환 기록\n  - 반품/교환한 상품\n  - 사유\n  - 처리 결과\n\n**3. 웹사이트 및 앱 활동 데이터**\n- 페이지 방문 기록\n  - 방문한 페이지\n  - 체류 시간\n  - 클릭한 링크\n- 검색 기록\n  - 검색어\n  - 검색 시간\n  - 검색 결과 클릭 여부\n- 장바구니 활동\n  - 장바구니에 추가한 상품\n  - 제거한 상품\n  - 보관 기간\n\n**4. 고객 서비스 상호작용 데이터**\n- 고객 문의 기록\n  - 문의 내용\n  - 문의 채널\n  - 처리 상태\n- 피드백 및 리뷰\n  - 제품 및 서비스에 대한 평가\n  - 리뷰 내용\n  - 평점\n\n**5. 소셜 미디어 데이터**\n- 소셜 미디어 상호작용\n  - 좋아요\n  - 공유\n  - 댓글\n- 브랜드 언급\n  - 브랜드에 대한 언급 내용\n  - 감정 분석 결과\n\n**6. 위치 데이터**\n- 오프라인 매장 방문 기록\n  - 방문한 매장\n  - 방문 시간\n  - 구매 여부\n- 위치 기반 서비스 이용 기록\n  - GPS를 통한 위치 추적 데이터\n\n**7. 기타 데이터**\n- 설문 조사 응답\n  - 고객 만족도 조사 결과\n  - 선호도 조사 응답\n- 로그인 및 인증 기록\n  - 로그인 시간\n  - 로그인 방법\n  - 실패 시도\n\n— \n\n**< 예시 >**\n- 고객 프로필 데이터: <연령: 35, 성별: 여성, 지역: 서울, 직업: 회사원>\n- 거래 데이터: <구매한 상품: TV, 구매 날짜: 2021-05-20, 구매 금액: 500,000원, 결제 수단: 신용카드>\n- 웹사이트 및 앱 활동 데이터: <방문한 페이지: 상품 카테고리, 체류 시간: 10분, 클릭한 링크: 할인 이벤트>\n- 고객 서비스 상호작용 데이터: <문의 내용: 반품 문의, 문의 채널: 이메일, 처리 상태: 완료>\n- 소셜 미디어 데이터: <좋아요: 100개, 공유: 50회, 댓글: \"좋은 제품입니다.\">\n- 위치 데이터: <방문한 매장: 이마트 강남점, 방문 시간: 오후 3시, 구매 여부: 있음>\n- 기타 데이터: <고객 만족도 조사 결과: 매우 만족, 선호도 조사 응답: 가격, 품질>\n\n4. AI 도입 프로세스\n**이마트와 The Home Depot의 AI 기반 고객 행동 분석 시스템 구현 프로세스:**\n\n1. 파일럿 프로젝트 준비\n   - 목표 설정 및 비즈니스 요구 분석\n   - 데이터 수집 및 준비\n   - 기술 및 도구 선정\n\n2. 파일럿 프로젝트 실행\n   - 모델 개발 및 학습\n   - 테스트 및 검증\n   - 성과 평가\n\n3. 전체 도입 계획 수립\n   - 확장 계획 수립\n   - 자원 및 인프라 준비\n\n4. 전체 도입 및 운영\n   - 시스템 구축 및 통합\n   - 교육 및 변화 관리\n   - 지속적인 모니터링 및 개선\n\n5. 예상 비용 및 ROI\n- 예상 평균 비용 범위\n    - 데이터 수집 및 전처리: 수백만 원에서 수천만 원\n    - 모델 개발 및 학습: 60만 원에서 200만 원\n    - 시스템 통합 및 배포: 수백만 원\n    - 유지보수 및 업데이트: 수백만 원에서 수백만 원\n    \n- 예상 ROI\n    - 고객 만족도 향상, 매출 증대, 고객 이탈률 감소, VIP 고객 재구매율 증가로 인한 매출 성장을 고려할 때, 예상 ROI는 100% 이상으로 예상됩니다.\n\n"
test_result_summary = "**1. 경쟁사 AI 도입 사례**\n- **카맥스:**\n  - 목적: 고객 리뷰 요약 생성을 통한 고객 구매 결정 지원\n  - 성과: 수천 개의 차량 페이지 리뷰 요약으로 고객 경험 향상\n  - 영향: 고객 구매 결정 속도 향상 및 구매로 이어질 가능성 증가\n\n- **스티치 픽스:**\n  - 목적: AI를 활용한 맞춤화된 의류 추천\n  - 성과: 트렌드 분석과 요구사항 식별로 효율적인 재고 관리와 로지스틱스 최적화\n  - 영향: 고객 만족도 향상 및 비용 절감\n\n- **브이캣:**\n  - 목적: 쇼핑몰 URL 입력으로 광고 소재 생성을 돕는 AI 솔루션 제공\n  - 성과: 300% 이상 증가한 ROAS 및 광고 성과 향상\n  - 영향: 빠른 광고 소재 제작으로 광고 효과 향상\n\n- **레코픽:**\n  - 목적: 고객 행동 데이터 분석을 통한 개인화된 상품 추천 제공\n  - 성과: 매출 15.6% 증가 및 구매 건수 16.7% 증가\n  - 영향: 고객 구매 전환율 향상 및 매출 증대 기여\n\n**2. Pain point 관련 추천 AI**\n- **서비스 설명:** 강화 학습을 활용한 고객 행동 예측 시스템\n- **기대 효과:** 정확한 행동 분석으로 이탈률 감소, VIP 고객 재구매율 증가 및 구매 전환율 향상\n\n**3. AI 도입 시 필요 데이터**\n- **고객 행동 예측 시스템을 위한 데이터 목록:** 고객 프로필, 거래, 활동, 서비스 상호작용, 소셜 미디어, 위치, 기타 데이터\n\n**4. AI 도입 프로세스**\n- **RAG 데이터를 활용한 강화 학습을 활용한 고객 행동 예측 시스템 구현 프로세스:** 파일럿 프로젝트 준비, 실행, 전체 도입 계획 수립, 전체 도입 및 운영\n\n**5. 예상 비용 및 ROI**\n- **예상 평균 비용 범위:** 데이터 수집, 모델 개발, 시스템 통합, 유지보수에 대한 총 예산 범위\n- **예상 ROI:** 150% ~ 200%"

data_id = 0;

app = FastAPI()
class ConsultingRequest(BaseModel):
    industry: Industry = Field(..., description="산업군 (Retail)")
    company_size: CompanySize = Field(..., description="기업 규모 (Small, Medium, Large)")
    pain_point: str = Field(..., description="기업 내에서 겪고 있는 문제점")
    class Config:
        schema_extra = {
            "example": {
                "industry": "Retail",
                "company_size": "Medium",
                "pain_point": "매출 증대를 위한 고객 행동 분석"
            }
        }


class ConsultingResponse(BaseModel):
    result: str = Field(..., description="컨설팅 결과 내용")


@app.post("/api/consulting",
          summary="컨설팅 결과 생성",
          description="주어진 정보를 바탕으로 컨설팅 결과를 생성하고, 필요하면 요약된 결과를 반환",
          response_description="컨설팅 결과 내용",
          response_model=ConsultingResponse
          )

async def get_consulting_result(request: ConsultingRequest,
    summary: bool = Query(False, description="컨설팅 결과 정보 요약 여부 (True/False)"),  # 쿼리 파라미터 추가
    test: bool = Query(False, description="임시 테스트 데이터 요청 (True/False)")
):
    if test is True:
        if summary is True:
            return {"result": test_result_summary}
        else:
            return {"result": test_result}

    global data_id
    data_id += 1
    industry = request.industry
    company_size = request.company_size
    pain_point = request.pain_point

    response = await create_consulting_result(industry=industry, company_size=company_size, pain_point=pain_point, data_id=data_id);
    if summary is True:
        response = get_summary_result(response)
    return {"result": response}

@app.websocket("/ws")
async def websocket_connect(websocket: WebSocket):
    await websocket.accept()
    connected_clients.append(websocket)
    try:
        while True:
            await asyncio.sleep(10)  # 10초마다 Heartbeat 전송
    except WebSocketDisconnect:
        connected_clients.remove(websocket)
        print("클라이언트 연결 해제")

@app.websocket("/ws/test")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()

    connected_clients.append(websocket)

    await send_info_test()





# if __name__ == '__main__':

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
    # get_all_data()
    # get_saved_doc_data_by_id(ES_INDEX_NAME, "ySAcLpMBhs7LHBg3Nqdm")
    # response = create_consulting_result(industry=industry, company_size=company_size, pain_point="매출 증대를 위한 고객 행동 분석.")
    # print(response)

