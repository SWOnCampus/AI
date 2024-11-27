import json
from globals import connected_clients
import asyncio
from fastapi import WebSocketDisconnect

lock = asyncio.Lock()

async def send_info(id, title, data_type, content):
    # 전송할 데이터 예시
    data = {
        'id': id,
        'title': title,
        'data_type': data_type,
        'content': content
    }

    remove_list = [];

    for client in connected_clients:
        try:
            await client.send_text(json.dumps(data, ensure_ascii=False))
        except WebSocketDisconnect as e:
            remove_list.append(client)
            continue


    for remove in remove_list:
        async with lock:
            connected_clients.remove(remove)

async def send_info_test():
    while True:
        for client in connected_clients:
            await client.send_text(f"data")