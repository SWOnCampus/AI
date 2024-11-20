import pdfplumber
import openai
import os
import requests
from constants import MAAL_BASE_URL, MAAL_APP_ID, MAAL_NAME, MAAL_ITEM
from prompts import GENERATE_QNA_LLM_SYSTEM_PROMPT

# OpenAI API 키
openai.api_key = os.getenv('OPENAI_API_KEY')

# PDF 파일 경로 설정
pdf_path = '../src/sample.pdf'

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
    url = MAAL_BASE_URL,

    # 바디 설정
    body = {
        "app_id": MAAL_APP_ID,
        "name": MAAL_NAME,
        "item": [
            MAAL_ITEM
        ],
        "param": [
            {
                "utterances": [
                    {
                        "role": "ROLE_SYSTEM",
                        "content": f"{GENERATE_QNA_LLM_SYSTEM_PROMPT}"
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
    text_prompt = f"""
텍스트 내용:
{text}
"""
    prompt =  GENERATE_QNA_LLM_SYSTEM_PROMPT + text_prompt

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

