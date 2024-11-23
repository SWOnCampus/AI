import constants
from storage.elastic_search_storage import Category, get_similar_qna_data
from hallucination.validate_hallucination import validate_hallucination
from .generate_consulting_prompt import *
from log.send_log_data import send_info
import openai
import re


OPENAI_MODEL = "gpt-3.5-turbo"
OPENAI_MAX_TOKENS = 4096

PRINT_REQUEST_PROMPT = False # 프롬프트 출력 여부

CONSULTING_RESULT_TITLES = ["1. 경쟁사 AI 도입 사례", "2. Pain Point 관련 추천 AI", "3. AI 도입 시 필요 데이터", "4. AI 도입 프로세스", "5. 예상 비용 및 ROI"] # 컨설팅 내용 별 타이틀 목록

SEND_LOG_SOCKET = True # 웹 소켓 로그 전달 여부

openai.api_key = constants.OPENAI_API_KEY

# 컨설팅 데이터 생성
async def create_consulting_result(industry, company_size, pain_point, data_id):
    result_contents = []
    result_contents.append(get_result_content(title=CONSULTING_RESULT_TITLES[0], content=await get_industry_example_content(industry=industry, company_size=company_size, data_id= data_id))) # AI 사례

    recommend_ai_answer, ai_service = await get_ai_service_datas(pain_point=pain_point, industry=industry, company_size=company_size, data_id=data_id)

    result_contents.append(get_result_content(title=CONSULTING_RESULT_TITLES[1], content=recommend_ai_answer)) # 추천 AI 서비스

    data_category_rag = get_category_rag_data(ai_service= ai_service, industry=industry)

    result_contents.append(get_result_content(title=CONSULTING_RESULT_TITLES[2], content= await get_ai_service_require_data(industry=industry, ai_service=ai_service, rag_data=data_category_rag, data_id=data_id))) # AI 도입 시 필요 데이터

    result_contents.append(get_result_content(title=CONSULTING_RESULT_TITLES[3],
                       content= await get_ai_service_process(ai_service=ai_service, data_category_rag=data_category_rag, industry=industry, company_size=company_size, data_id=data_id))) # AI 도입 프로세스

    result_contents.append(get_result_content(title=CONSULTING_RESULT_TITLES[4],
                       content= await get_ai_service_ROI(ai_service=ai_service, industry=industry, company_size=company_size, recommend_ai_answer=recommend_ai_answer, data_id=data_id))) # 예상 비용 및 ROI

    return "".join(result_contents)




# 경쟁사 AI 도입 사례 내용 생성
async def get_industry_example_content(industry, company_size, data_id):
    rag_data = merge_industry_example(industry=industry, company_size=company_size, category=Category.Industry)

    await send_info(id=data_id, title=1, data_type="rag", content=rag_data)

    prompt = generate_industry_examples_prompt(industry=industry, company_size=company_size, examples=rag_data)

    return await get_LLM_answer_with_validate(prompt=prompt, rag_data=rag_data, title=1, data_id=data_id)



# pain point 기반 추천 AI 서비스
async def get_recommend_ai_service(pain_point, industry, company_size, retry, data_id):
    response = get_similar_qna_data(data_size=5, company_size=company_size, category=Category.PainPoints,
                                    industry=industry, question=pain_point)

    rag_data = extract_and_merge_answer(response)

    await send_info(id=data_id, title=2, data_type="rag", content=rag_data)

    prompt = generate_ai_service_prompt(pain_point=pain_point, industry=industry, company_size=company_size,
                                        examples=rag_data, retry=retry)
    return await get_LLM_answer_with_validate(prompt=prompt, rag_data=rag_data, title=2, data_id=data_id)


# AI 서비스 관련 필요 데이터
async def get_ai_service_require_data(industry, ai_service, rag_data, data_id):
    await send_info(id=data_id, title=3, data_type="rag", content=rag_data)

    prompt = generate_ai_service_require_data_prompt(industry=industry, ai_service=ai_service, rag_data=rag_data)

    return await get_LLM_answer_with_validate(prompt=prompt, rag_data=rag_data, title=3, data_id=data_id)


# AI 서비스 구축 프로세스
async def get_ai_service_process(ai_service, data_category_rag, industry, company_size, data_id):

    response = get_similar_qna_data(data_size=1, question=ai_service, category=Category.Process)

    rag_data = extract_and_merge_answer(response)

    await send_info(id=data_id, title=4, data_type="rag", content=rag_data)

    prompt = generate_ai_service_process_prompt(company_size=company_size, industry=industry, ai_service=ai_service,
                                                rag_data=rag_data, data_category_rag_data=data_category_rag)

    return await get_LLM_answer_with_validate(prompt, rag_data + data_category_rag, title=4, data_id=data_id)

# AI 서비스 관련 예상 ROI
async def get_ai_service_ROI(ai_service, industry, company_size, recommend_ai_answer, data_id):
    response = get_similar_qna_data(data_size=1, question=ai_service, category=Category.CostROI,
                                    company_size=company_size, industry=industry)

    rag_data = extract_and_merge_answer(response)

    await send_info(id=data_id, title=5, data_type="rag", content=rag_data)

    prompt = generate_ai_service_ROI_prompt(company_size=company_size, industry=industry, ai_service=ai_service,
                                            recommend_ai_answer=recommend_ai_answer, rag_data=rag_data)

    return await get_LLM_answer_with_validate(prompt=prompt, rag_data=rag_data + ai_service, title=5, data_id=data_id)


# AI 컨설팅 데이터 요약
async def get_summary_result(result, data_id):
    prompt = generate_summary_result_prompt(result)

    await send_info(id=data_id, title=30, data_type="summary", content=result)

    return await get_LLM_answer_with_validate(prompt=prompt, rag_data=result, title=30, data_id=data_id)

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


# RAG 데이터 합쳐서 반환
def extract_and_merge_answer(response):
    source = [hit["_source"]["answer"] for hit in response["hits"]["hits"]]
    examples = ""

    for example in source:
        examples += example + "\n"

    return examples

# 경쟁사 AI 도입 사례 프롬프트 추가 내용
def merge_industry_example(industry, company_size, category):
    response = get_similar_qna_data(data_size=5, company_size= company_size, category= category, industry= industry)

    return extract_and_merge_answer(response)



# "AI 서비스" 항목 추출
def get_ai_service_from_answer(answer):

    match = re.search(r"AI 서비스\s*:\s*(.+)", answer)

    if match:
        ai_service = match.group(1).strip()
        return ai_service
    else:
        print("AI 서비스 정보를 찾을 수 없습니다.")


# 추천 AI 서비스 관련 응답 내용, 추출된 AI 응답 서비스
async def get_ai_service_datas(pain_point, industry, company_size, data_id):
    answer = None # 추천 AI 서비스 LLM 응답 결과
    extract_ai_service = None # 응답 결과에서 추출 된 추천 AI 서비스 (LLM이 응답 형식에 맞게 답변하는 경우 추출)
    retry = False # 추천 AI 서비스 추출 여부

    while extract_ai_service is None: # 만약 ai_service를 추출할 수 없으면 다시 요청
        answer = await get_recommend_ai_service(pain_point= pain_point, industry=industry, company_size=company_size, retry= retry, data_id=data_id)
        extract_ai_service = get_ai_service_from_answer(answer)
        retry = True # 재 요청 여부를 True로 변환, LLM 요청 프롬프트에서 형식 중요 추가

    return answer, extract_ai_service

# 데이터 카테고리 관련 RAG 데이터 추출
def get_category_rag_data(ai_service, industry):
    data_category_rag_datas = get_similar_qna_data(data_size=1, question=ai_service, category=Category.Data, industry=industry)
    data_category_rag = extract_and_merge_answer(data_category_rag_datas)

    return data_category_rag

# 할루시네이션 검증을 포함하여 LLM 답변
async def get_LLM_answer_with_validate(prompt, rag_data, title, data_id, temperature=0.7):
    while(True):
        answer = get_chat_gpt_answer(prompt=prompt, temperature=temperature)

        await send_info(id=data_id, title=title, data_type="answer", content=answer)

        if(await validate_hallucination(rag_data=rag_data, answer=answer, title=title, data_id=data_id) is True):
            return answer


# 컨설팅 결과 데이터 내용 추가
def get_result_content(title, content):
    return f"{title}\n{content}\n\n"