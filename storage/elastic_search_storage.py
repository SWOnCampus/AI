import numpy as np
import os
from embeddings.sentence_transform import sentence_embedding
from elasticsearch import Elasticsearch
from enum import Enum
from constants import ES_PASSWORD, ES_USERNAME, ES_BASE_URL, ES_INDEX_NAME

# cert 파일 경로
ES_CA_CERT_PATH = os.path.join(os.path.dirname(__file__), "..", "cert", "http_ca.crt")

# Elasticsearch 클라이언트 생성
es = Elasticsearch(
    [ES_BASE_URL],
    http_auth=(ES_USERNAME, ES_PASSWORD),
    verify_certs=True,
    ca_certs=ES_CA_CERT_PATH # http_ca.crt 파일 경로
)

class CompanySize(Enum):
    MICRO = "Micro"  # 초소형 기업
    SMALL = "Small"  # 소기업
    MEDIUM = "Medium"  # 중견 기업
    LARGE = "Large"  # 대기업

def create_database():

    # json 데이터 정보 매핑
    schema = get_database_schema()

    es.indices.create(index=ES_INDEX_NAME, body=schema, ignore = 400) # 이미 존재하는 경우 무시

def get_database_schema():
    mapping = {
        "mappings": {
            "properties": {
                "company_size": {"type": "text"},
                "question": {"type": "text"},
                "answer": {"type": "text"},
                "embedding": {
                    "type": "dense_vector",
                    "dims": 512  # 임베딩 벡터의 차원 수에 맞게 설정
                }
            }
        }
    }

    return mapping


def save_qna_data(company_size, question, answer):

    # 질문 내용 기준으로 임베딩
    embeddings = sentence_embedding(question)

    # 벡터 정규화
    question_embedding = normalize_embeddings(embeddings)

    # Elastic_search 데이터 스키마 형식으로 변경
    doc = create_doc_data(company_size, question, answer, question_embedding)

    response = es.index(index=ES_INDEX_NAME, document=doc)

    print("저장된 데이터 : ", response)

def create_doc_data(company_size, question, answer, question_embedding):
    doc = {
        "company_size": company_size.value,
        "question": question,
        "answer": answer,
        "embedding": question_embedding
    }

    return doc

def normalize_embeddings(embeddings):
    norms = np.linalg.norm(embeddings, axis=0, keepdims=True)
    return embeddings / norms


def get_saved_doc_data_by_id(index_name, doc_id):
    try:
        response = es.get(index=index_name, id= doc_id)
        print("문서 데이터:", response['_source'])  # 문서의 내용 출력
    except Exception as e:
        print("오류 발생:", e)

def get_similar_qna_data(company_size, ):


def create_query(query_vector, company_size, data_size):
    # 쿼리 작성
    query = {
        "size": data_size,  # 검색할 인접 벡터 수
        "query": {
            "bool": {
                "filter": [
                    {"term": {"company_size": company_size.value}},  # 필터 조건 기업 규모에 맞게
                ],
                "must": {
                    "script_score": {
                        "query": {"match_all": {}},
                        "script": {
                            "source": "cosineSimilarity(params.query_vector, 'vector_field') + 1.0", # 코사인 유사도로 검색
                            "params": {"query_vector": query_vector}
                        }
                    }
                }
            }
        }
    }

    return query