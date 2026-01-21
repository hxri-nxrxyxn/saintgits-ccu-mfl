def make_server_collection_id(session_id: str) -> str:
    return session_id

def make_client_collection_id(session_id: str, client_id: str) -> str:
    return session_id + "-" + client_id
