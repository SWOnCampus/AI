import json
from globals import connected_clients

async def send_info(id, title, data_type, content):
    # 전송할 데이터 예시
    data = {
        'id': id,
        'title': title,
        'data_type': data_type,
        'content': content
    }


    for client in connected_clients:
        try:
            await client.send_text(json.dumps(data, ensure_ascii=False))
        except Exception as e:
            connected_clients.remove(client)
            continue;


async def send_info_test():
    while True:
        for client in connected_clients:
            await client.send_text(f"data")