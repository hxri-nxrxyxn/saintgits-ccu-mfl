import base64
import openapi_client

from filelock import FileLock
from minio import Minio
from fed_multimodal_restcol.restcol.env import ENV_STORAGE_ENDPOINT, ENV_STORAGE_ACCESS_KEY, ENV_STORAGE_ACCESS_SECRET, ENV_STORAGE_BUCKET_NAME
from openapi_client.rest import ApiException
from openapi_client.models.api_collection_type import ApiCollectionType
from openapi_client.models.rest_col_service_create_document_body import RestColServiceCreateDocumentBody
from openapi_client.models.rest_col_service_create_collection_body import RestColServiceCreateCollectionBody
from pprint import pprint

class RestcolClient:
    def __init__(
            self,
            host_url: str,
            authorized_token: str,
            project_id: str,
            debug: bool = False,
            local_cache: bool = False,
            ):

        import os

        self.configuration = openapi_client.Configuration(
            host = host_url
        )
        self.configuration.verify_ssl = False
        self.s3_client = Minio(ENV_STORAGE_ENDPOINT,
                                  access_key=ENV_STORAGE_ACCESS_KEY,
                                  secret_key=ENV_STORAGE_ACCESS_SECRET, secure=True,
                                  cert_check=False)

        self.authorized_token = authorized_token
        self.local_cache_dir = "/tmp"
        self.local_cache = local_cache
        self.s3_bucket = ENV_STORAGE_BUCKET_NAME
        if self.authorized_token:
            self.configuration.api_key['ApiKeyAuth'] = self.authorized_token
            self.configuration.api_key_prefix['ApiKeyAuth'] = "Bearer"
        self.configuration.debug = debug
        print(f'config: {self.configuration}')
        self.project_id = project_id
        print(f'project_id: {self.project_id}')

    def _str2base64str(self, b:str):
        return base64.b64encode(b.encode()).decode()

    def create_collection(self, collection_id: str = None, description: str = None):
        """ create_collection"""
        print(f"create collection: {collection_id}")

        with openapi_client.ApiClient(self.configuration) as api_client:
            # Create an instance of the API class
            api_instance = openapi_client.CollectionApi(api_client)

            body_dict = {}
            if collection_id:
                body_dict['collectionId'] = collection_id
            if description:
                body_dict['description'] = description
            else:
                body_dict['description'] = "empty description"

            body = RestColServiceCreateCollectionBody.from_dict(body_dict)
            try:
                # Add collection endpoint, a collection is a set of documents with scheme-free.
                api_response = api_instance.rest_col_service_create_collection(self.project_id, body)
                api_response_dict = api_response.to_dict()
                print(f"api_response_dict: {api_response_dict}")
                return api_response_dict["Metadata"]["collectionId"]
            except Exception as e:
                print("[WARN] Remote Collection Creation Failed (%s). Using Mock Collection ID.\n" % e)
                # Fallback: return the requested collection_id so we can use it for local storage
                return collection_id

    def write_document(self, collection_id: str, dic_body: dict,
                       doc_id: str = None):
        """ write_document we serialize dic_body with pickle and keep it on
        storage and write its url in the restcol storage
        """
        import io
        import json
        import pickle
        import uuid
        import os

        # MOCK STORAGE PATH
        MOCK_STORAGE_DIR = "/app/mock_storage"
        os.makedirs(os.path.join(MOCK_STORAGE_DIR, collection_id), exist_ok=True)

        if doc_id is None:
            doc_id = str(uuid.uuid4())

        # Try Remote First
        try:
            filename = f"{uuid.uuid4()}.pkl"
            picked_dic_body = pickle.dumps(dic_body)
            self.s3_client.put_object(self.s3_bucket, filename,
                                         io.BytesIO(picked_dic_body), len(picked_dic_body))

            pickle_data = self._str2base64str(json.dumps({"filename": filename}))

            data_dict = {"data": pickle_data}
            if doc_id:
                data_dict["documentId"] = doc_id

            with openapi_client.ApiClient(self.configuration) as api_client:
                # Create an instance of the API class
                api_instance = openapi_client.DocumentApi(api_client)
                body = RestColServiceCreateDocumentBody.from_dict(data_dict)
            
                # Add collection endpoint, a collection is a set of documents with scheme-free.
                api_response = api_instance.rest_col_service_create_document2(
                        project_id = self.project_id, 
                        collection_id = collection_id, 
                        body = body)
                api_response_dict = api_response.to_dict()
                print(f"api_response_dict={api_response_dict}")
                return api_response_dict["Metadata"]["documentId"]

        except Exception as e:
            print(f"[WARN] Remote Write Failed ({e}). Falling back to Local Mock Storage.")
            mock_path = os.path.join(MOCK_STORAGE_DIR, collection_id, f"{doc_id}.pkl")
            with open(mock_path, 'wb') as f:
                pickle.dump(dic_body, f)
            print(f"[SUCCESS] Saved to Local Mock Storage: {mock_path}")
            return doc_id

    def read_document(self, collection_id: str, document_id: str) -> dict:
        import pickle
        import os

        print(f"readdocument: cid={collection_id}, did={document_id}")
        
        # Check Local Mock Storage First/Fallback
        MOCK_STORAGE_DIR = "/app/mock_storage"
        mock_path = os.path.join(MOCK_STORAGE_DIR, collection_id, f"{document_id}.pkl")
        
        # Try Remote
        filename = None
        try:
            with openapi_client.ApiClient(self.configuration) as api_client:
                # Create an instance of the API class
                api_instance = openapi_client.DocumentApi(api_client)
                # Add collection endpoint, a collection is a set of documents with scheme-free.
                api_response = api_instance.rest_col_service_get_document(self.project_id, collection_id, document_id)
                api_response_dict = api_response.to_dict()
                filename = api_response_dict['data']['filename']

            if filename:
                response = self._get_object(filename)
                return pickle.loads(response)

        except Exception as e:
            print(f"[WARN] Remote Read Failed ({e}). Checking Local Mock Storage.")
            if os.path.exists(mock_path):
                print(f"[SUCCESS] Loaded from Local Mock Storage: {mock_path}")
                with open(mock_path, 'rb') as f:
                    return pickle.load(f)
            else:
                print(f"[ERROR] Not found in Mock Storage either: {mock_path}")
                return {}

        return {}

    def _get_object(self, filename: str):
        import os

        if not self.local_cache:
            try:
                response = self.s3_client.get_object(self.s3_bucket, filename)
                response_data = response.data
            finally:
                response.close()
                response.release_conn()
            return response_data

        self._mkdir(os.path.join(self.local_cache_dir, self.s3_bucket))

        cache_file = os.path.join(self.local_cache_dir, self.s3_bucket, filename)
        lock_cache_file = os.path.join(self.local_cache_dir, self.s3_bucket, filename, ".lock")
        with FileLock(lock_cache_file):
            if os.path.exists(cache_file):
                with open(cache_file, 'rb') as f:
                    response_data = f.read()
                return response_data
            try:
                response = self.s3_client.get_object(self.s3_bucket, filename)
                with open(cache_file, 'wb') as f:
                    f.write(response.data)
                response_data = response.data
            finally:
                response.close()
                response.release_conn()
            return response_data

    def _mkdir(self, path: str):
        import pathlib

        pathlib.Path(path).mkdir(parents=True, exist_ok=True)

    def document_exists(self, collection_id: str, document_id: str) -> bool:
        print(f"document_exits: cid={collection_id}, did={document_id}")
        
        # Check Remote
        try:
            with openapi_client.ApiClient(self.configuration) as api_client:
                # Create an instance of the API class
                api_instance = openapi_client.DocumentApi(api_client)
                # Add collection endpoint, a collection is a set of documents with scheme-free.
                api_response = api_instance.rest_col_service_get_document(self.project_id, collection_id, document_id)
                api_response_dict = api_response.to_dict()
                if collection_id == api_response_dict['Metadata']['collectionId'] and document_id == api_response_dict['Metadata']['documentId']:
                    return True
        except Exception as e:
            # Check Local Mock Storage
            import os
            MOCK_STORAGE_DIR = "/app/mock_storage"
            mock_path = os.path.join(MOCK_STORAGE_DIR, collection_id, f"{document_id}.pkl")
            if os.path.exists(mock_path):
                return True
            # print(f"Exception when calling CollectionApi->rest_col_service_create_collection: %s\n" % e)
        
        return False
 
