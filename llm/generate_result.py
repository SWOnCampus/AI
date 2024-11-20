from storage.elastic_search_storage import Category, get_similar_qna_data
from hallucination.validate_hallucination import validate_hallucination
from .generate_consulting_prompt import *
import openai
import re

OPENAI_MODEL = "gpt-3.5-turbo"
OPENAI_MAX_TOKENS = 4096

PRINT_REQUEST_PROMPT = False # 프롬프트 출력 여부

# 경쟁사 AI 도입 사례 내용 생성
def get_industry_example_content(industry, company_size):
    while (True):
         examples = merge_industry_example(industry=industry, company_size=company_size, category=Category.Industry)

         prompt = generate_industry_examples_prompt(industry=industry, company_size=company_size, examples=examples)
         answer = get_chat_gpt_answer(prompt)

         if validate_hallucination(examples, answer) is True:
             return answer

def get_chat_gpt_answer(prompt, temperature=0.7):
    if PRINT_REQUEST_PROMPT is True:
        print(prompt)

    response = openai.ChatCompletion.create(
        model=OPENAI_MODEL,
        messages=[{'role': 'user', 'content': prompt}],
        max_tokens=OPENAI_MAX_TOKENS,
        temperature=temperature,
    )

    return response['choices'][0]['message']['content']

# 경쟁사 AI 도입 사례 프롬프트 추가 내용
def merge_industry_example(industry, company_size, category):
    response = get_similar_qna_data(data_size=5, company_size= company_size, category= category, industry= industry)

    return extract_and_merge_answer(response)


def extract_and_merge_answer(response):
    source = [hit["_source"]["answer"] for hit in response["hits"]["hits"]]
    examples = ""

    for example in source:
        examples += example + "\n"

    return examples

# pain point 기반 추천 AI 서비스
def get_recommend_ai_service(pain_point, industry, company_size, retry):
    while(True):
        response = get_similar_qna_data(data_size=5, company_size=company_size, category=Category.PainPoints, industry= industry, question=pain_point)

        examples = extract_and_merge_answer(response)

        prompt = generate_ai_service_prompt(pain_point=pain_point, industry=industry, company_size=company_size, examples= examples, retry= retry)

        answer = get_chat_gpt_answer(prompt=prompt, temperature=0.3)

        if validate_hallucination(examples, answer) is True:
             return answer

        return answer;


def get_ai_service_from_answer(answer):
    # 정규식을 사용해 "AI 서비스" 항목 추출
    match = re.search(r"AI 서비스\s*:\s*(.+)", answer)

    if match:
        ai_service = match.group(1).strip()
        return ai_service
    else:
        print("AI 서비스 정보를 찾을 수 없습니다.")

def create_consulting_result(industry, company_size, pain_point):
    result = "1. 경쟁사 AI 도입 사례\n"
    result += get_industry_example_content(industry=industry, company_size=company_size)

    ai_service = None
    retry = False

    recommend_ai_answer = None;

    while ai_service is None: # 만약 ai_service를 추출할 수 없으면 다시 요청
        recommend_ai_answer = get_recommend_ai_service(pain_point= pain_point, industry=industry, company_size=company_size, retry= retry)
        ai_service = get_ai_service_from_answer(recommend_ai_answer)
        retry = True # 재 요청 여부를 True로 변환, LLM 요청 프롬프트에서 형식 중요 추가

    result += "\n\n2. Pain Point 관련 추천 AI\n"
    result += recommend_ai_answer

    response = get_similar_qna_data(data_size=1, question=ai_service, category=Category.Data, industry=industry)
    data_category_rag = extract_and_merge_answer(response)

    result += "\n\n3. AI 도입 시 필요 데이터\n"
    result += get_ai_service_require_data(industry, ai_service, data_category_rag) # 필요 데이터 내용 생성

    result += "\n\n4. AI 도입 프로세스\n"
    result += get_ai_service_process(ai_service=ai_service, data_category_rag=data_category_rag, industry=industry, company_size=company_size)

    result += "\n\n5. 예상 비용 및 ROI\n"
    result += get_ai_service_ROI(ai_service=ai_service, industry=industry, company_size=company_size, recommend_ai_answer=recommend_ai_answer)

    return result

def get_ai_service_require_data(industry, ai_service, rag_data):
    while(True):
        prompt = generate_ai_service_require_data_prompt(industry=industry, ai_service=ai_service, rag_data=rag_data)

        answer = get_chat_gpt_answer(prompt)

        if validate_hallucination(rag_data, answer) is True:
             return answer

        return answer

def get_ai_service_process(ai_service, data_category_rag, industry, company_size):
    while(True):
        response = get_similar_qna_data(data_size=1, question=ai_service, category=Category.Process)

        rag_data = extract_and_merge_answer(response)

        prompt = generate_ai_service_process_prompt(company_size=company_size, industry=industry, ai_service=ai_service, rag_data= rag_data, data_category_rag_data=data_category_rag)

        answer = get_chat_gpt_answer(prompt)

        if validate_hallucination(rag_data + data_category_rag, answer) is True:
             return answer

        return answer

def get_ai_service_ROI(ai_service, industry, company_size, recommend_ai_answer):
    while(True):
        response = get_similar_qna_data(data_size=1, question=ai_service, category=Category.CostROI, company_size=company_size, industry=industry)

        rag_data = extract_and_merge_answer(response)

        prompt = generate_ai_service_ROI_prompt(company_size=company_size, industry=industry, ai_service=ai_service, recommend_ai_answer=recommend_ai_answer, rag_data=rag_data)

        answer = get_chat_gpt_answer(prompt)

        if validate_hallucination(ai_service + rag_data, answer) is True:
             return answer

        return answer


def get_summary_result(result):

    while(True):

        prompt = generate_summary_result_prompt(result)

        answer = get_chat_gpt_answer(prompt)

        if validate_hallucination(result, answer) is True:
             return answer

        return answer






# merge_industry_example(industry=Industry.Retail, company_size=CompanySize.MEDIUM, category=Category.Industry)
# get_industry_example_content(Industry.Retail, CompanySize.MEDIUM)
# answer = get_recommend_ai_service(pain_point="매출 증대를 위한 고객 행동 분석", industry=Industry.Retail, company_size=CompanySize.MEDIUM, retry=False)
# get_ai_service_require_data(Industry.Retail, "딥러닝")

# ai_service = "매출 증대를 위한 고객 행동 분석"
# response = get_similar_qna_data(data_size=1, question=ai_service, category=Category.Data, industry=Industry.Retail)
# data_category_rag = extract_and_merge_answer(response)
# get_ai_service_process(ai_service, data_category_rag, Industry.Retail, company_size=CompanySize.MEDIUM)
# get_ai_service_ROI(ai_service=ai_service, industry=Industry.Retail, company_size=CompanySize.MEDIUM, recommend_ai_answer=answer)


a = """
지마켓은 AI Product 팀을 통해 개인화 추천 기술과 서비스를 개발하였습니다. 특히 모바일 앱의 홈 화면을 개인화하는 프로젝트를 진행하여, 고객당 클릭률이 이전 대비 40% 향상되고, 고객들이 클릭한 상품 수가 2배 이상 증가하는 성과를 거두었습니다. 이를 통해 구매자와 판매자 모두의 만족도를 개선하고, 모바일 홈에서의 매출 증가를 이끌어냈습니다.
레코픽은 고객의 행동 데이터를 분석하여 개인화된 상품 추천을 제공하는 AI 솔루션을 개발하였습니다. 이를 통해 평균 매출이 15.6% 증가하고, 구매 건수가 16.7% 상승하는 등의 성과를 달성하였습니다. 이러한 AI 기반 추천 시스템은 고객의 구매 전환율을 높이는 데 크게 기여하였습니다.
브이캣은 쇼핑몰 URL을 입력하면 자동으로 텍스트, 이미지, 동영상 광고 소재를 생성해주는 AI 솔루션을 제공합니다. 이를 통해 다양한 광고 소재를 빠르게 제작하여 테스트할 수 있으며, 평균 300% 증가한 ROAS를 기록하였습니다. 일부 광고 소재는 1,000% 이상의 성과를 보이기도 하였습니다.
SAP의 조사에 따르면, 국내 중견기업 중 매출 성장률이 높은 기업일수록 생성형 AI 도입을 비즈니스의 우선순위로 고려하고 있습니다. 매출 성장률이 높은 기업의 96%가 생성형 AI 도입을 '보통' 또는 '높은' 우선순위로 인식하고 있으며, 이를 통해 고객 경험 혁신, 데이터 보안 강화, 교육 및 개발 등 다양한 분야에서 AI를 활용하고 있습니다.
온라인 쇼핑 회사인 스티치 픽스는 고객의 스타일 선호도를 수집하고 이를 스타일리스트의 전문 지식과 AI 기술과 결합하여 개인에게 맞춤화된 의류를 추천합니다. 이 회사는 AI를 통해 패션 트렌드를 분석하고, 고객의 변화하는 요구사항을 식별하여 효율적인 재고 관리와 로지스틱스를 최적화합니다.
"""


b = """
**지마켓**
- **AI 도입 현황:** 개인화 추천 기술과 서비스를 개발하였으며, 모바일 앱의 홈 화면을 개인화하는 프로젝트를 진행함.
- **구체적인 성과:**
  - 고객당 클릭률이 이전 대비 40% 향상.
  - 고객들이 클릭한 상품 수가 2배 이상 증가.
- **영향:** 구매자와 판매자 모두의 만족도를 개선하고, 모바일 홈에서의 매출 증가를 이끔.

**레코픽**
- **AI 도입 현황:** 고객의 행동 데이터를 분석하여 개인화된 상품 추천을 제공하는 AI 솔루션을 개발함.
- **구체적인 성과:**
  - 평균 매출이 15.6% 증가.
  - 구매 건수가 16.7% 상승.
- **영향:** 고객의 구매 전환율을 높이는 데 기여.

**브이캣**
- **AI 도입 현황:** 쇼핑몰 URL을 입력하면 자동으로 광고 소재를 생성해주는 AI 솔루션을 제공함.
- **구체적인 성과:**
  - 평균 300% 증가한 ROAS.
  - 일부 광고 소재는 1,000% 이상의 성과를 보임.
- **영향:** 다양한 광고 소재를 빠르게 제작하여 테스트할 수 있으며, 광고 효율을 대폭 향상시킴.
"""


result = validate_hallucination(a, b);
