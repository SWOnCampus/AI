from storage.elastic_search_storage import Industry, Category, get_similar_qna_data, CompanySize
import openai
import re

# 경쟁사 AI 도입 사례 내용 생성
def get_industry_example_content(industry, company_size):

     examples = merge_industry_example(industry=industry, company_size=company_size, category=Category.Industry)

     prompt = f"""
다음은 {industry.get_string()} 산업 분야의 {company_size.get_string()}들이 AI를 도입한 사례들입니다. 제공된 데이터만을 사용하여 경쟁사의 AI 도입 사례를 전문적이고 명확하게 정리해 주세요. 추가적인 정보나 추측 없이, 주어진 데이터에 기반하여 내용을 구성해야 합니다. PDF 보고서에 포함될 수 있도록 구조화된 형식으로 작성해 주세요. 각 기업의 이름, AI 도입 목적, 구체적인 성과 및 영향을 포함해 주세요.
---

**RAG 데이터:**
{examples}

---

**요청 사항:**
- 각 기업의 AI 도입 사례를 명확하고 간결하게 요약해 주세요.
- 전문적인 어조를 유지하며, 각 사례를 개별 섹션으로 구분해 주세요.
- 성과를 구체적인 숫자와 함께 강조해 주세요.
- 추가적인 정보나 추측 없이 제공된 데이터만을 사용해 주세요.
- PDF 보고서에 적합하도록 깔끔하고 읽기 쉬운 형식으로 작성해 주세요.

---

**답변 형식:**

**실제 기업 명**
- **AI 도입 현황:**
- **구체적인 성과:**
  -
  -
- **영향:**
"""
     answer = get_chat_gpt_answer(prompt)

     print(answer)

     return answer

def get_chat_gpt_answer(prompt, temperature=0.7):
    response = openai.ChatCompletion.create(
        model='gpt-3.5-turbo',
        messages=[{'role': 'user', 'content': prompt}],
        max_tokens=4096,
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
    response = get_similar_qna_data(data_size=5, company_size=company_size, category=Category.PainPoints, industry= industry, question=pain_point)

    examples = extract_and_merge_answer(response)

    prompt = f"""
다음은 {industry.get_string()} 산업 분야의 {company_size.get_string()}들이 문제점을 해결하기 위해 AI를 도입한 사례들입니다. 제공된 데이터만을 사용하여 문제점을 해결하기 위해 가장 적합한 AI서비스를 추천해주세요. 추가적인 정보나 추측 없이, 주어진 데이터에 기반하여 내용을 구성해야 합니다. PDF 보고서에 포함될 수 있도록 구조화된 형식으로 작성해 주세요. 추천하는 AI기술과 기술에 대한 설명, 기대효과를 구체적이고 명확하게 정리해 주세요.
---
**문제점**
{pain_point}

---
**RAG 데이터:**
{examples}

---
**요청 사항:**
- 추천 AI 서비스에 대한 설명과 기대 효과를 도입 사례들을 기반으로 작성해 주세요. 
- AI 서비스에는 사용되는 "핵심 AI 기술의 이름"을 포함하여 작성해 주세요.
- 추가적인 정보나 추측 없이 제공된 데이터만을 사용해 주세요.
- PDF 보고서에 적합하도록 깔끔하고 읽기 쉬운 형식으로 작성해 주세요.
{'- **답변 형식에 맞게 답변해 주세요.**' if retry else ''}

---
**답변 형식**
AI 서비스 :
서비스에 대한 설명 : 
기대 효과 :
"""
    print(prompt)

    answer = get_chat_gpt_answer(prompt=prompt, temperature=0.3)

    print(answer)

    return answer;


def get_ai_service_from_answer(answer):
    # 정규식을 사용해 "AI 서비스" 항목 추출
    match = re.search(r"AI 서비스\s*:\s*(.+)", answer)

    if match:
        ai_service = match.group(1).strip()
        print(ai_service)
        return ai_service
    else:
        print("AI 서비스 정보를 찾을 수 없습니다.")

def create_consultion_result(industry, company_size, pain_point):
    result = "1. 경쟁사 AI 도입 사례\n"

    result += get_industry_example_content(industry=industry, company_size=company_size)

    ai_service = None
    retry = False

    while ai_service is None: # 만약 ai_service를 추출할 수 없으면 다시 요청
        recommend_ai_answer = get_recommend_ai_service(pain_point= pain_point, industry=industry, company_size=company_size, retry= retry)
        ai_service = get_ai_service_from_answer(recommend_ai_answer)
        retry = True # 재 요청 여부를 True로 변환, LLM 요청 프롬프트에서 형식 중요 추가

    response = get_similar_qna_data(data_size=1, question=ai_service, category=Category.Data, industry=industry)

    data_category_rag = extract_and_merge_answer(response)

    result += get_ai_service_require_data(industry, ai_service, data_category_rag) # 필요 데이터 내용 생성

    result += get_ai_service_process(ai_service=ai_service, data_category_rag=data_category_rag, industry=industry, company_size=company_size)


def get_ai_service_require_data(industry, ai_service, rag_data):

    prompt = f"""
다음은 {industry.get_string} 기업에서 {ai_service}를 구현하기 위해 필요한 데이터 목록입니다. 제공된 RAG 데이터를 이용해서 필요한 데이터 목록들을 전문적이고 명확하게 정리해 주세요. 추가적인 정보나 추측 없이, 주어진 데이터에 기반하여 내용을 구성해야 합니다. PDF 보고서에 포함될 수 있도록 구조화된 형식으로 작성해 주세요. 필요한 데이터의 대주제, 소주제, 예시들을 포함해 주세요.
---
**RAG 데이터:**
{rag_data}

---
**요청 사항:**
- 추가적인 정보나 추측 없이 제공된 데이터만을 사용해 주세요.
- PDF 보고서에 적합하도록 깔끔하고 읽기 쉬운 형식으로 작성해 주세요.
- 답변 형식은 개조식으로, 데이터의 주제 하위에 데이터의 예시가 포함되도록 작성해 주세요.
"""

    print(prompt)

    answer = get_chat_gpt_answer(prompt)

    print(answer)
    return answer

def get_ai_service_process(ai_service, data_category_rag, industry, company_size):
    response = get_similar_qna_data(data_size=1, question=ai_service, category=Category.Process)

    rag_data = extract_and_merge_answer(response)

    prompt = f"""
다음은 {company_size.get_string} {industry.get_string}에서 {ai_service}를 구현하기 위한 프로세스 과정과 활용할 데이터 입니다. 제공된 RAG 데이터들을 이용해서 {ai_service} 구현 프로세스를 전문적이고 명확하게 정리해 주세요. 추가적인 정보나 추측 없이, 주어진 데이터에 기반하여 내용을 구성해야 합니다. PDF 보고서에 포함될 수 있도록 구조화된 형식으로 작성해 주세요.
---
**RAG 데이터 - 프로세스:**
{rag_data}

---
**RAG 데이터 - 활용 데이터:**
{data_category_rag}

---
**요청 사항:**
- 추가적인 정보나 추측 없이 제공된 데이터만을 사용해 주세요.
- PDF 보고서에 적합하도록 깔끔하고 읽기 쉬운 형식으로 작성해 주세요.
- 답변은 순차적으로, 어떤 과정이 필요한지 자세하게 답변해 주세요.
- 주어진 기업 규모와 산업 종류에 맞춰서 프로세스를 작성해 주세요.
- 답변은 개조식으로 작성해주세요.
"""
    print(prompt)

    answer = get_chat_gpt_answer(prompt)

    print(answer)
    return answer

# merge_industry_example(industry=Industry.Retail, company_size=CompanySize.MEDIUM, category=Category.Industry)
# get_industry_example_content(Industry.Retail, CompanySize.MEDIUM)
# get_recommend_ai_service(pain_point="매출 증대를 위한 고객 행동 분석", industry=Industry.Retail, company_size=CompanySize.MEDIUM, retry=True)
# get_ai_service_require_data(Industry.Retail, "딥러닝")

# ai_service = "매출 증대를 위한 고객 행동 분석"
# response = get_similar_qna_data(data_size=1, question=ai_service, category=Category.Data, industry=Industry.Retail)
# data_category_rag = extract_and_merge_answer(response)
# get_ai_service_process(ai_service, data_category_rag, Industry.Retail, company_size=CompanySize.MEDIUM)
