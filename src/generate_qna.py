import pdfplumber
import openai
import os
import requests

# OpenAI API 키
openai.api_key = os.getenv('OPENAI_API_KEY')

# PDF 파일 경로 설정
pdf_path = 'sample.pdf'

# PDF에서 텍스트 추출
def extract_text_from_pdf(pdf_path):
    text = ''
    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text + '\n'
    return text

# 텍스트를 max_length 기준으로 분할
def split_text(text, max_length=10000):
    parts = []
    while len(text) > max_length:
        # 마지막 공백에서 자르기 (단어 최대한 살리기 위해)
        split_index = text.rfind(' ', 0, max_length)
        if split_index == -1:
            split_index = max_length
        parts.append(text[:split_index])
        text = text[split_index:].strip()
    parts.append(text)
    return parts

def generate_qna(parts, num_pairs):
    all_qna = []
    for part in parts:
        qna = generate_qna_maal(part, num_pairs)
        all_qna.extend(qna)

    return all_qna

# maal를 사용하여 Q&A 생성
def generate_qna_maal(text, num_pairs=5):
    # 헤더 설정
    headers = {
        "Content-Type": "application/json",
        "Cache-Control": "no-cache"
    }

    # URL 설정
    url = "https://norchestra.maum.ai/harmonize/dosmart"

    # 바디 설정
    body = {
        "app_id": "4b2a80e0-23bd-58f8-bd93-dc170681839c",
        "name": "hansung_70b_chat",
        "item": [
            "maumgpt-maal2-70b-chat"
        ],
        "param": [
            {
                "utterances": [
                    {
                        "role": "ROLE_SYSTEM",
                        "content": f"""
                                    **사용자가 PDF 파일에서 추출한 텍스트를 AI 기술을 도입하려는 기업의 컨설팅 내용을 Q&A 형식으로 만들어주세요. AI 기술을 도입하려는 기업의 예상 질문과 입력된 텍스트의 내용을 기반으로 작성된 답변을 만들어 주세요.**

                                    중요 사항:
                                    입력 텍스트는 PDF 파일에서 추출된 것이므로, 불필요한 형식이나 정보는 무시하고 핵심 내용에 집중하여 응답해 주세요.
                                    답변 예시를 참고하여 답변을 작성해주세요. 답변의 내용은 예시의 내용이 아닌 사용자의 입력에 대한 내용으로만 작성해야 합니다.
                                    Q&A 응답 데이터의 개수는 정확도를 보장할 수 있는 최대한의 개수만큼 만들어주세요.


                                    답변 예시:
                                    Q1: 챗봇을 도입하여 사용자 경험을 어떻게 향상시킬 수 있나요?
                                    A1: 24/7 고객 지원: 언제든지 고객 문의에 응답하여 고객 만족도를 높입니다.
                                        비용 절감: 인건비를 절감하고, 반복적인 업무를 자동화하여 운영 효율성을 향상시킵니다.
                                        응답 속도 향상: 실시간으로 신속하게 응답하여 고객 대기 시간을 줄입니다.
                                        데이터 수집 및 분석: 고객과의 대화를 통해 유용한 데이터를 수집하고, 이를 분석하여 비즈니스 전략에 활용할 수 있습니다.
                                        고객 경험 개선: 개인화된 응답과 추천을 통해 고객 경험을 향상시킵니다.
                                        업셀링 및 크로스셀링: 고객의 요구에 맞는 제품이나 서비스를 추천하여 매출을 증대시킬 수 있습니다.

                                    Q2: What are the most commonly reported uses of generative AI tools?
                                    A2: The most commonly reported uses of generative AI tools are in marketing and sales, product and service development, and service operations.


                                    Q3: 데이터 분석을 위해 AI를 도입하면 어떤 주요 이점을 얻을 수 있나요?
                                    A3: 데이터 분석 AI 도입의 주요 이점은 다음과 같습니다:
                                        정확한 예측: 과거 데이터를 기반으로 미래의 트렌드와 패턴을 정확하게 예측할 수 있습니다.
                                        의사결정 지원: 데이터 기반의 인사이트를 제공하여 더 나은 비즈니스 의사결정을 지원합니다.
                                        운영 효율성 향상: 데이터를 분석하여 비효율적인 프로세스를 식별하고 최적화할 수 있습니다.
                                        고객 이해 증진: 고객의 행동과 선호도를 분석하여 개인화된 마케팅 전략을 수립할 수 있습니다.
                                        리스크 관리: 잠재적인 리스크를 사전에 예측하고 대응 전략을 마련할 수 있습니다.
                                        혁신 촉진: 데이터 분석을 통해 새로운 비즈니스 기회를 발견하고 혁신을 촉진할 수 있습니다.

                                    Q4: What are some of the potential benefits of using generative AI tools?
                                    A4: Some of the potential benefits of using generative AI tools include crafting first drafts of text documents, identifying trends in customer needs, using chatbots for customer service, summarizing text documents, creating new product designs, and forecasting service trends or anomalies.
                                    """
                    },
                    {
                        "role": "ROLE_USER",
                        "content": f"{text}"
                    }
                ],
                "config": {
                    "top_p": 0.8,
                    "top_k": 5,
                    "temperature": 0.7,
                    "presence_penalty": 0.0,
                    "frequency_penalty": 0.0,
                    "repetition_penalty": 1.0
                }
            }
        ]
    }

    # API 호출
    response = requests.post(url, headers=headers, json=body)

    if response.status_code == 200:
        response_json = response.json()
        if response_json.get("finish_reason") != "stop":
            print(response_json.get("finish_reason"))
            print("MAAL 응답 에러")
        else:
            result = response_json.get("text", "")
            print(":::: :::: :::: :::: :::: MAAL 응답 내용 :::: :::: :::: :::: ::::")
            print(result)
            return result

    # 오류 발생 시
    else:
        print("Error:", response.status_code)
        print("Response Body:", response.text)



# GPT를 사용하여 Q&A 생성
def generate_qna_gpt(text, num_pairs=5):
    print(text)

    prompt = f"""다음 텍스트를 기반으로 {num_pairs}개의 질문과 그에 대한 답변을 만들어 주세요.

텍스트:
\"\"\"
{text}
\"\"\"

형식:
Q1: 질문 내용
A1: 답변 내용

Q2: 질문 내용
A2: 답변 내용

...

주의사항:
- 질문은 텍스트의 핵심 내용을 포함해야 합니다.
- 답변은 정확하고 간결하게 작성해주세요.
"""

    response = openai.ChatCompletion.create(
        model='gpt-3.5-turbo',
        messages=[{'role': 'user', 'content': prompt}],
        max_tokens=1500,
        temperature=0.7,
    )

    return response['choices'][0]['message']['content']

# 메인 실행 부분
if __name__ == '__main__':
    # PDF에서 텍스트 추출
    extracted_text = extract_text_from_pdf(pdf_path)
    if not extracted_text.strip():
        print("PDF에서 텍스트를 추출하지 못했습니다.")
    else:
        # Q&A 생성
        # qa_output = generate_qna_gpt(extracted_text, num_pairs=5)
        # qa_output = generate_qna_maal(extracted_text, num_pairs=5)
        parts = split_text(extracted_text, 15000)
        qa_output = generate_qna(parts, 5)

