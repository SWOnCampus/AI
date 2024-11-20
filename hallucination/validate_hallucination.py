from bert_score import score

def validate_hallucination(rag_data, answer):

    print("::::: 입력 RAG 데이터 :::::\n")
    print(rag_data + "\n\n")

    print("::::: LLM 반환 내용 :::::\n")
    print(answer + "\n\n")

    # BERTScore 계산
    P, R, F1 = score([answer], [rag_data], lang="ko", verbose=False)

    # 결과 출력
    print(f"Precision: {P.mean().item():.4f}")
    print(f"Recall: {R.mean().item():.4f}")
    print(f"F1 Score: {F1.mean().item():.4f}")

    # print(f":::: 할루시네이션 검증 유사도 : {similarity * 100}% ::::\n")

    if(F1.mean().item() >= 0.6):
        return True
    else:
        return False

