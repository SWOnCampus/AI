from storage.elastic_search_storage import Category, get_similar_qna_data
from hallucination.validate_hallucination import validate_hallucination
from .generate_consulting_prompt import *
import openai
import re

OPENAI_MODEL = "gpt-3.5-turbo"
OPENAI_MAX_TOKENS = 4096

PRINT_REQUEST_PROMPT = False # 프롬프트 출력 여부

CONSULTING_RESULT_TITLES = ["1. 경쟁사 AI 도입 사례", "2. Pain Point 관련 추천 AI", "3. AI 도입 시 필요 데이터", "4. AI 도입 프로세스", "5. 예상 비용 및 ROI"]

# 컨설팅 데이터 생성
def create_consulting_result(industry, company_size, pain_point):
    result = ""
    append_result_content(result=result, title=CONSULTING_RESULT_TITLES[0], content=get_industry_example_content(industry=industry, company_size=company_size)) # AI 사례

    recommend_ai_answer = None # 추천 AI 서비스 LLM 응답 결과
    ai_service = None # 응답 결과에서 추출 된 추천 AI 서비스 (LLM이 응답 형식에 맞게 답변하는 경우 추출)
    retry = False # 추천 AI 서비스 추출 여부

    while ai_service is None: # 만약 ai_service를 추출할 수 없으면 다시 요청
        recommend_ai_answer = get_recommend_ai_service(pain_point= pain_point, industry=industry, company_size=company_size, retry= retry)
        ai_service = get_ai_service_from_answer(recommend_ai_answer)
        retry = True # 재 요청 여부를 True로 변환, LLM 요청 프롬프트에서 형식 중요 추가

    append_result_content(result=result, title=CONSULTING_RESULT_TITLES[1], content=recommend_ai_answer)

    data_category_rag_datas = get_similar_qna_data(data_size=1, question=ai_service, category=Category.Data, industry=industry)
    data_category_rag = extract_and_merge_answer(data_category_rag_datas)


    append_result_content(result=result, title=CONSULTING_RESULT_TITLES[2], content=get_ai_service_require_data(industry, ai_service, data_category_rag))

    append_result_content(result=result, title=CONSULTING_RESULT_TITLES[3],
                          content=get_ai_service_process(ai_service=ai_service, data_category_rag=data_category_rag, industry=industry, company_size=company_size))

    append_result_content(result=result, title=CONSULTING_RESULT_TITLES[4],
                          content=get_ai_service_ROI(ai_service=ai_service, industry=industry, company_size=company_size, recommend_ai_answer=recommend_ai_answer))

    return result

# 경쟁사 AI 도입 사례 내용 생성
def get_industry_example_content(industry, company_size):
    while (True):
         examples = merge_industry_example(industry=industry, company_size=company_size, category=Category.Industry)

         prompt = generate_industry_examples_prompt(industry=industry, company_size=company_size, examples=examples)
         answer = get_chat_gpt_answer(prompt)

         if validate_hallucination(examples, answer) is True:
             return answer

# 경쟁사 AI 도입 사례 프롬프트 추가 내용
def merge_industry_example(industry, company_size, category):
    response = get_similar_qna_data(data_size=5, company_size= company_size, category= category, industry= industry)

    return extract_and_merge_answer(response)

# RAG 데이터 합쳐서 반환
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


# AI 서비스 관련 필요 데이터
def get_ai_service_require_data(industry, ai_service, rag_data):
    while(True):
        prompt = generate_ai_service_require_data_prompt(industry=industry, ai_service=ai_service, rag_data=rag_data)

        answer = get_chat_gpt_answer(prompt)

        if validate_hallucination(rag_data, answer) is True:
             return answer

        return answer


# AI 서비스 구축 프로세스
def get_ai_service_process(ai_service, data_category_rag, industry, company_size):
    while(True):
        response = get_similar_qna_data(data_size=1, question=ai_service, category=Category.Process)

        rag_data = extract_and_merge_answer(response)

        prompt = generate_ai_service_process_prompt(company_size=company_size, industry=industry, ai_service=ai_service, rag_data= rag_data, data_category_rag_data=data_category_rag)

        answer = get_chat_gpt_answer(prompt)

        if validate_hallucination(rag_data + data_category_rag, answer) is True:
             return answer

        return answer

# AI 서비스 관련 예상 ROI
def get_ai_service_ROI(ai_service, industry, company_size, recommend_ai_answer):
    while(True):
        response = get_similar_qna_data(data_size=1, question=ai_service, category=Category.CostROI, company_size=company_size, industry=industry)

        rag_data = extract_and_merge_answer(response)

        prompt = generate_ai_service_ROI_prompt(company_size=company_size, industry=industry, ai_service=ai_service, recommend_ai_answer=recommend_ai_answer, rag_data=rag_data)

        answer = get_chat_gpt_answer(prompt)

        if validate_hallucination(ai_service + rag_data, answer) is True:
             return answer

        return answer


# AI 컨설팅 데이터 요약
def get_summary_result(result):
    while(True):
        prompt = generate_summary_result_prompt(result)
        answer = get_chat_gpt_answer(prompt)

        if validate_hallucination(result, answer) is True:
             return answer

        return answer


# CHAT_GPT_응답 생성
def get_chat_gpt_answer(prompt, temperature=0.7):
    #프롬프트 출력 여부
    if PRINT_REQUEST_PROMPT is True:
        print(prompt)

    response = openai.ChatCompletion.create(
        model=OPENAI_MODEL,
        messages=[{'role': 'user', 'content': prompt}],
        max_tokens=OPENAI_MAX_TOKENS,
        temperature=temperature,
    )

    return response['choices'][0]['message']['content']

# "AI 서비스" 항목 추출
def get_ai_service_from_answer(answer):

    match = re.search(r"AI 서비스\s*:\s*(.+)", answer)

    if match:
        ai_service = match.group(1).strip()
        return ai_service
    else:
        print("AI 서비스 정보를 찾을 수 없습니다.")


# 컨설팅 결과 데이터 내용 추가
def append_result_content(result, title, content):
    result += f"{title}\n{content}\n\n"