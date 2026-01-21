import numpy as np
import pytest

from deepdiff import DeepDiff
from fed_multimodal_restcol.restcol.client import RestcolClient

host = "http://host.docker.internal:50091"

def test_restcol_client_create_document():

    default_collection_id = "0192bc97-7292-701b-a11a-1c8041438ebc"

    data_dict = {
        "array_fields": np.zeros((10,2)),
        "dict_fields": {
            "key1": "value1",
            "key2": 123456,
        }
    }

    ## running inside container
    c = RestcolClient(host_url = host, authorized_token = "", project_id = "1001")
    document_id = c.write_document(default_collection_id, data_dict)

    recv_data_dict = c.read_document(default_collection_id, document_id)
    print(recv_data_dict)

    assert bool(DeepDiff(data_dict, recv_data_dict)) == False

def test_restcol_client_read_document_not_found():
    default_collection_id = "0192bc97-7292-701b-a11a-1c8041438ebc"

    ## running inside container
    c = RestcolClient(host_url = host, authorized_token = "", project_id = "1001")
    document_exists = c.document_exists(default_collection_id, "should-have-no-document")
    assert document_exists == False


test_restcol_client_create_document()
test_restcol_client_read_document_not_found()
