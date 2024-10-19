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

    print(text)
    return text

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
                        "content": f"사용자가 입력한 텍스트를 기업 컨설팅 내용을 기반으로 {num_pairs}개의 질문과 그에 대한 답변을 만들어 주세요. 질문과 답변은 다음 내용을 중심으로 작성해주세요:\n - 기업의 이름은 빼고 작성해주세요.\n - 질문과 응답은 기업 내의 AI Transformation을 중심으로 작성해주세요. \n\n답변 형식:\nQ1: 질문 내용\nA1: 답변 내용\n\nQ2: 질문 내용\nA2: 답변 내용\n\n\n주의 사항:\n- 질문은 텍스트의 핵심 내용을 포함해야 합니다.\n- 답변은 정확하고 간결하게 작성해주세요."
                    },
                    {
                        "role": "ROLE_USER",
                        "content": f"{text}"
                    }
                ],
                "stream": False
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

    print(openai.api_key)

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
        qa_output = generate_qna_maal(extracted_text, num_pairs=5)
        print("생성된 Q&A:")
        print(qa_output)
