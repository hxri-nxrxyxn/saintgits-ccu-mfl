import argparse
import os
import pickle

from fed_multimodal_restcol.restcol.client import RestcolClient

def parse_args():

    parser = argparse.ArgumentParser(
        description='FedMultimodal upload dataset to restcol'
    )

    parser.add_argument(
        '--collection_id',
        type=str,
        help="collection_id for holding data",
    )

    parser.add_argument(
        '--restcol_host',
        default="http://host.docker.internal:50091",
        type=str,
        help="host url for restcol service",
    )

    parser.add_argument(
        '--pkls',
        action='store',
        nargs='+',
        type=str,
        help="pkls for upload",
    )

    parser.add_argument(
        '--restcol_authtoken',
        default='',
        type=str,
        help='restcol auth token'
    )

    parser.add_argument(
        '--restcol_projectid',
        default='',
        type=str,
        help='restcol project id'
    )



    args = parser.parse_args()

    return args


def run():
    args = parse_args()
    print(f"args = {args}")
    restcol_client = RestcolClient(host_url = args.restcol_host,
                                   authorized_token = args.restcol_authtoken,
                                   project_id = args.restcol_projectid)

    collection_id = restcol_client.create_collection(args.collection_id, "collection for holding dataset")

    print(f"pkls={args.pkls}" )
    data_dict = {}
    for pkl in args.pkls:
        base_name = os.path.basename(pkl)
        with open(pkl, 'rb') as f:
            obj = pickle.load(f)
        data_dict[base_name] = obj

    for key in data_dict:
        print(f"key = {key}")

    doc_id = restcol_client.write_document(collection_id, data_dict)
    print(f"docid: {doc_id}")


if __name__ == '__main__':
    run()

