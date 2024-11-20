from fastapi import FastAPI, Query
from pydantic import BaseModel, Field
from embeddings.sentence_transform import sentence_embedding
from storage.faiss_storage import sentence_embedding_save, search_similar_sentences
from storage.elastic_search_storage import create_database, save_qna_data, CompanySize, get_similar_qna_data, get_saved_doc_data_by_id, get_all_data, Category, Industry
from llm.generate_result import create_consulting_result, get_summary_result
from constants import ES_INDEX_NAME

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

def get_consulting_result(request: ConsultingRequest,
    summary: bool = Query(False, description="컨설팅 결과 정보 요약 여부 (True/False)")  # 쿼리 파라미터 추가
):
    industry = request.industry
    company_size = request.company_size
    pain_point = request.pain_point

    response = create_consulting_result(industry=industry, company_size=company_size, pain_point=pain_point);
    if summary is True:
        response = get_summary_result(response)
    return {"result": response}



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



