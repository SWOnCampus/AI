from typing import List
from fastapi import WebSocket

connected_clients: List[WebSocket] = []
