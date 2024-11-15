from storage.elastic_search_storage import Industry, CompanySize, Category, get_similar_qna_data
import openai

RECOMMEND_AI_PROMPT = """
아래의 실제 AI도입 예시를 이용해서 문제점을 해결하기 위해 가장 적합한 AI 기술과 그에 대한 설명, 그리고 AI 도입을 통한 기대 효과를 다음과 같은 형식으로 하나만 답변해주세요.

내용은 모두 서술식으로 작성하세요.

문제점 : 매출 증대를 위한 고객 행동 분석

<답변 형식>

AI 기술 :
기술에 대한 설명 : 
기대 효과 :

<AI 도입 예시>
"""


def get_industry_example_content(industry, company_size):
     prompt = f"""
아래의 실제 AI 활용 사례를 설명 글로 다듬어서 자세하게 작성해주세요.

산업 분야 : {industry.get_string()}
기업 규모 : {company_size.get_string()}
AI 도입 사례 :
"""

     prompt += merge_industry_example(industry= industry, company_size= company_size, category=Category.Industry)

     response = openai.ChatCompletion.create(
         model='gpt-3.5-turbo',
         messages=[{'role': 'user', 'content': prompt}],
         max_tokens=4096,
         temperature=0.7,
     )


     answer = response['choices'][0]['message']['content']
     print(answer)

     return answer

def merge_industry_example(industry, company_size, category):
    response = get_similar_qna_data(data_size=5, company_size= company_size, category= category, industry= industry)
    source = [hit["_source"]["answer"] for hit in response["hits"]["hits"]]
    examples = ""

    for example in source:
        examples += example + "\n"

    return examples

# merge_industry_example(industry=Industry.Retail, company_size=CompanySize.MEDIUM, category=Category.Industry)
get_industry_example_content(Industry.Retail, CompanySize.MEDIUM)