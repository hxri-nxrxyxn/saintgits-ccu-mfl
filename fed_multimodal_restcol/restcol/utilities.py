import time
from fed_multimodal_restcol.restcol.client import RestcolClient

class timeout(object):
    def __init__(self, seconds):
        self.seconds = seconds
    def __enter__(self):
        self.die_after = time.time() + self.seconds
        return self
    def __exit__(self, type, value, traceback):
        pass
    @property
    def timed_out(self):
        return time.time() > self.die_after


def wait_doc_exists_until(client: RestcolClient, collection_id: str, document_id: str, timeout_in_sec: int) -> bool:
    print(f"wait doc exist until: cid={collection_id}, did={document_id}")
    with timeout(timeout_in_sec) as t:
        while True:
            if t.timed_out:
                break

            if client.document_exists(collection_id, document_id):
                print("wait_doc_exists_until.exist, exit")
                return True

            print("wait_doc_exists_until.10s")
            time.sleep(10) # sleep 10s for the next try

    return False




