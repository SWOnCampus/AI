def generate_industry_examples_prompt(industry, company_size, examples):
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

    return prompt


def generate_ai_service_prompt(pain_point, industry, company_size, examples, retry):
    prompt = f"""
다음은 {industry.get_string()} 산업 분야의 {company_size.get_string()}들이 문제점을 해결하기 위해 AI를 도입한 사례들입니다. 제공된 데이터만을 사용하여 문제점을 해결하기 위해 가장 적합한 AI서비스를 추천해주세요. 추가적인 정보나 추측 없이, 주어진 데이터에 기반하여 내용을 구성해야 합니다. PDF 보고서에 포함될 수 있도록 구조화된 형식으로 작성해 주세요. 추천하는 AI기술과 기술에 대한 설명, 기대효과를 구체적이고 명확하게 정리해 주세요.
---
**문제점**
{pain_point}

---
**RAG 데이터:**
{examples}

---
요청 사항:
- 추천 AI 서비스에 대한 설명과 기대 효과를 도입 사례들을 기반으로 작성해 주세요. 
- AI 서비스에는 사용되는 **핵심 AI 기술의 이름(ex. 딥러닝)**을 포함하여 작성해 주세요.
- 추가적인 정보나 추측 없이 제공된 데이터만을 사용해 주세요.
- PDF 보고서에 적합하도록 깔끔하고 읽기 쉬운 형식으로 작성해 주세요.
- 기대 효과는 pain point 해결과 관련되게 구체적으로 작성해주세요.
{'- **답변 형식에 맞게 답변해 주세요.**' if retry else ''}

---
답변 형식
AI 서비스 :
서비스에 대한 설명 : 
기대 효과 :
"""

    return prompt


def generate_ai_service_require_data_prompt(industry, ai_service, rag_data):
    prompt = f"""
다음은 {industry.get_string} 기업에서 {ai_service}를 구현하기 위해 필요한 데이터 목록입니다. 제공된 RAG 데이터를 이용해서 필요한 데이터 목록들을 전문적이고 명확하게 정리해 주세요. 추가적인 정보나 추측 없이, 주어진 데이터에 기반하여 내용을 구성해야 합니다. PDF 보고서에 포함될 수 있도록 구조화된 형식으로 작성해 주세요. 필요한 데이터의 대주제, 소주제, 예시들을 포함해 주세요.
---
**RAG 데이터:**
{rag_data}

---
**요청 사항:**
- 추가적인 정보나 추측 없이 제공된 데이터만을 사용해 주세요.
- PDF 보고서에 적합하도록 깔끔하고 읽기 쉬운 형식으로 작성해 주세요.
- 답변 형식은 개조식으로, 데이터의 주제 하위에 데이터의 예시(예: < 표기할 것)가 포함되도록 작성해 주세요.
"""

    return prompt


def generate_ai_service_process_prompt(company_size, industry, ai_service, rag_data, data_category_rag_data):
    prompt = f"""
다음은 {company_size.get_string} {industry.get_string}에서 {ai_service}를 구현하기 위한 프로세스 과정과 활용할 데이터 입니다. 제공된 RAG 데이터들을 이용해서 {ai_service} 구현 프로세스를 전문적이고 명확하게 정리해 주세요. 추가적인 정보나 추측 없이, 주어진 데이터에 기반하여 내용을 구성해야 합니다. PDF 보고서에 포함될 수 있도록 구조화된 형식으로 작성해 주세요.
---
**RAG 데이터 - 프로세스:**
{rag_data}

---
**RAG 데이터 - 활용 데이터:**
{data_category_rag_data}

---
**요청 사항:**
- 추가적인 정보나 추측 없이 제공된 데이터만을 사용해 주세요.
- PDF 보고서에 적합하도록 깔끔하고 읽기 쉬운 형식으로 작성해 주세요.
- 답변은 순차적으로, 어떤 과정이 필요한지 자세하게 답변해 주세요.
- 주어진 기업 규모와 산업 종류에 맞춰서 프로세스를 작성해 주세요.
- 답변은 개조식으로 작성해주세요.
- 개조식에 해당하지 내용만 작성하세요.
"""

    return prompt


def generate_ai_service_ROI_prompt(company_size, industry, ai_service, recommend_ai_answer, rag_data):
    prompt = f"""
다음은 {company_size.get_string} {industry.get_string}에서 {ai_service}를 구현할 때 예상 예산 책정 방법을 위한 RAG 데이터 입니다. 제공된 RAG 데이터들을 이용해서 {ai_service} 구현할 때 예상 예산 금액과 ROI를 예측해 주세요. 추가적인 정보나 추측 없이, 주어진 데이터에 기반하여 내용을 구성해야 합니다. PDF 보고서에 포함될 수 있도록 구조화된 형식으로 작성해 주세요.
---
**RAG 데이터 - 구현할 AI 서비스:**
{recommend_ai_answer}

---
**RAG 데이터 - 예산 책정 방법:**
{rag_data}

---
**요청 사항:**
- 추가적인 정보나 추측 없이 제공된 데이터만을 사용해 주세요.
- 주어진 기업 규모와 산업 종류, 구현할 AI 서비스를 참고하여 예상 예산을 책정해 주세요.
- PDF 보고서에 적합하도록 깔끔하고 읽기 쉬운 형식으로 작성해 주세요.
- 답변은 **개조식**으로 작성해주세요.
- 예상 ROI는 수치로 나타내 주세요.
- 답변 형식에 맞춰서 답변해 주세요.
---
**답변 형식:**
- 예상 평균 비용 범위
    - 비용 산정 방식
- 예상 ROI
"""

    return prompt

def generate_summary_result_prompt(result):
    prompt = f"""
다음은 컨설팅 정보에 대한 내용입니다. 제공된 내용을 중요한 정보들 로만 요약해 주세요. 추가적인 정보나 추측 없이, 주어진 정보에 기반하여 내용을 구성해야 합니다. PDF 보고서에 포함될 수 있도록 구조화된 형식으로 작성해 주세요.
---
**컨설팅 정보 내용**
{result}
---
**요청 사항:**
- 추가적인 정보나 추측 없이 제공된 데이터만을 사용해 주세요.
- PDF 보고서에 적합하도록 깔끔하고 읽기 쉬운 형식으로 작성해 주세요.
- 답변은 **개조식**으로 작성해 주세요.
- 답변 형식에 맞춰서 답변해 주세요.
---
**답변 형식:**

1. 경쟁사 AI 도입 사례
  -

2. Pain point 관련 추천 AI
  -
  
3. AI 도입 시 필요 데이터
  -

4. AI 도입 프로세스
  -
  
5. 예상 비용 및 ROI
  -
"""

    return prompt
