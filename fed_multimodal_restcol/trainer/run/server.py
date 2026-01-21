import argparse, logging
import copy, shutil, sys, os, pdb
import torch
import torch.nn as nn
import uuid
from fed_multimodal_restcol.restcol.client import RestcolClient
from fed_multimodal_restcol.restcol.utilities import wait_doc_exists_until
from fed_multimodal_restcol.trainer.run.util import make_server_collection_id, make_client_collection_id
from fed_multimodal_restcol.trainer.remote.remote_trainer import Server
from fed_multimodal_restcol.trainer.model.mm_models import ImageTextClassifier

def parse_args():

    parser = argparse.ArgumentParser(
        description='FedMultimodal server flags'
    )
    parser.add_argument(
        '--data_dir',
        default='/tmp',
        type=str,
        help='output dir'
    )

    parser.add_argument(
        "--dataset",
        type=str,
        default="crisis-mmd",
        help='data set name'
    )

    parser.add_argument(
        '--img_feat',
        default='mobilenet_v2',
        type=str,
        help="image feature name",
    )

    parser.add_argument(
        '--text_feat',
        default='mobilebert',
        type=str,
        help="text feature name",
    )

    parser.add_argument(
        "--alpha",
        type=float,
        default=1.0,
        help="alpha in direchlet distribution",
    )

    parser.add_argument(
        '--local_epochs',
        default=1,
        type=int,
        help="local epochs",
    )

    parser.add_argument(
        '--learning_rate',
        default=0.1,
        type=float,
        help="learning rate",
    )

    parser.add_argument(
        '--global_learning_rate',
        default=0.05,
        type=float,
        help="learning rate",
    )

    parser.add_argument(
        '--fed_alg',
        default='fed_avg',
        type=str,
        help="federated learning aggregation algorithm",
    )

    parser.add_argument(
        '--batch_size',
        default=4,
        type=int,
        help="training batch size",
    )

    parser.add_argument(
        "--missing_modality",
        type=bool,
        default=False,
        help="missing modality simulation",
    )

    parser.add_argument(
        "--en_missing_modality",
        dest='missing_modality',
        action='store_true',
        help="enable missing modality simulation",
    )

    parser.add_argument(
        "--missing_modailty_rate",
        type=float,
        default=0.5,
        help='missing rate for modality; 0.9 means 90%% missing'
    )

    parser.add_argument(
        "--missing_label",
        type=bool,
        default=False,
        help="missing label simulation",
    )

    parser.add_argument(
        "--en_missing_label",
        dest='missing_label',
        action='store_true',
        help="enable missing label simulation",
    )

    parser.add_argument(
        "--missing_label_rate",
        type=float,
        default=0.5,
        help='missing rate for modality; 0.9 means 90%% missing'
    )

    parser.add_argument(
        '--label_nosiy',
        type=bool,
        default=False,
        help='clean label or nosiy label')

    parser.add_argument(
        "--en_label_nosiy",
        dest='label_nosiy',
        action='store_true',
        help="enable label noise simulation",
    )

    parser.add_argument(
        '--label_nosiy_level',
        type=float,
        default=0.1,
        help='nosiy level for labels; 0.9 means 90% wrong'
    )

    parser.add_argument(
        '--modality', 
        type=str, 
        default='multimodal',
        help='modality type'
    )

    # --client_ids id1 id2
    parser.add_argument(
        '--session_id',
        type=str,
        help="universal session id to pair server and client",
    	default=str(uuid.uuid4()),
    )
    parser.add_argument('--client_ids', nargs='+', help='<Required> Set flag', required=True)
    parser.add_argument(
        '--num_classes',
        default=5,
        type=int,
        help="num of classes",
    )
    parser.add_argument(
        '--img_input_dims',
        default=128,
        type=int,
        help="image input dims",
    )
    parser.add_argument(
        '--text_input_dims',
        default=32,
        type=int,
        help="text input dims",
    )
    parser.add_argument(
        '--hid_size',
        type=int,
        default=64,
        help='RNN hidden size dim'
    )

    parser.add_argument(
        '--att',
        type=bool,
        default=False,
        help='self attention applied or not'
    )

    parser.add_argument(
        "--en_att",
        dest='att',
        action='store_true',
        help="enable self-attention"
    )

    parser.add_argument(
        '--att_name',
        type=str,
        default='multihead',
        help='attention name'
    )

    parser.add_argument(
        '--sample_rate',
        default=0.1,
        type=float,
        help="client sample rate",
    )

    parser.add_argument(
        '--num_epochs',
        default=300,
        type=int,
        help="total training rounds",
    )

    parser.add_argument(
        '--restcol_host',
        default="http://host.docker.internal:50091",
        type=str,
        help="host url for restcol service",
    )

    parser.add_argument(
        '--restcol_authtoken',
        default='eyJhbGciOiJSUzI1NiIsImtpZCI6ImM4MzExZmZmNWE3NjY5ZjRmOTNjMTMyODg3NmRkYTc3NjJmYTMyYjkifQ.eyJpc3MiOiJodHRwczovL2F1dGguc2FpbnRnaXRzMDEuZm9vdHByaW50LWFpLmNvbS9kZXgiLCJzdWIiOiJDaU5qYmoxa1pXMXZMRzkxUFZCbGIzQnNaU3hrWXoxbGVHRnRjR3hsTEdSalBXOXlaeElFYkdSaGNBIiwiYXVkIjoia2FmZWlkby1hcHAiLCJleHAiOjE3NjM2MTUzMjYsImlhdCI6MTc2MzUyODkyNiwiYXRfaGFzaCI6InB2Rmd5cEEtdHhiZzBLUVJ2eWh3WEEiLCJlbWFpbCI6ImRlbW9AZm9vdHByaW50LWFpLmNvbSIsImVtYWlsX3ZlcmlmaWVkIjp0cnVlLCJncm91cHMiOlsic3AxMCJdLCJuYW1lIjoiZGVtbyJ9.TgE5DE1Y8zUFgVXmX4vPiwH09Xl_RkAMPNtELm_Om1Jn_f3z5OWAfi7i4ELk3sMOFUEQ-ZvFkQR7N5IiQ9Q8zTydJYcMWYsmxirivbMPqEmNyYYLTjg-mbgqcYRgeLI40Tod90KEPu0X7F73Brrk_15EVovB1ZLuCXVdYoiaej0nJ2OmtxhLfXhzKzLIkbjEUTlkFp5yhaT02Xv4Ii1cy7_pPBCjm-DR986UFEBsvNxIX9187REVkNuw1vhpiBA4uOvHn9s0IcN4QGfmA2IScVy2HxAlwjuznF5avt5NUK-iDNBFLXg1kKViqw0-CgFkVU-jQbtryFnhhm2af_0ang',
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
    print(f"args={args}")

    restcol_client = RestcolClient(host_url = args.restcol_host,
                                   authorized_token = args.restcol_authtoken,
                                   project_id = args.restcol_projectid,
                                   )

    device = torch.device("cpu") # Forced CPU for GT710 optimization
    if torch.cuda.is_available(): 
        print('GPU available, but using CPU per config')


    # for server side, train a global model
    # loss function
    criterion = nn.NLLLoss().to(device)
    # Define the model
    global_model = ImageTextClassifier(
        num_classes=args.num_classes,
        img_input_dim=args.img_input_dims,
        text_input_dim=args.text_input_dims,
        d_hid=args.hid_size,
        en_att=args.att,
        att_name=args.att_name
    )
    global_model = global_model.to(device)

    server = Server(
        args,
        global_model,
        device=device,
        criterion=criterion,
        client_ids=args.client_ids
    )
    server.initialize_log(1)
    server.sample_clients(len(args.client_ids), sample_rate=args.sample_rate)
    #print(f'server_collection_id={server_collection_id}')
    server.get_num_params()
	
    server_collection_id = make_server_collection_id(args.session_id)
    print(f'server_collection_id={server_collection_id}') # Added for debugging

    restcol_client.create_collection(server_collection_id, "server collection binded to session id")

    for epoch in range(int(args.num_epochs)):
        print(f'server.epoch: {epoch}')
        epoch_file = f"epoch-{epoch}"
        server.initialize_epoch_updates(epoch)

        restcol_client.write_document(server_collection_id, {'global_model':global_model}, epoch_file)

        # wait for client model
        client_updates_list = []
        for client_id in args.client_ids:
            client_epoch_file = f"{client_id}-epoch-{epoch}"
            wait_doc_exists_until(restcol_client, server_collection_id, client_epoch_file, 1000)
            client_updates = restcol_client.read_document(server_collection_id,
                                                          client_epoch_file)
            # client_updates = {
            #  "model": client.get_parameters(),
            #  "samples": int,
            #  "result": [],
            #  "delta_control": object,
            # }
            server.save_train_updates(
                copy.deepcopy(client_updates['model']),
                client_updates['sample'],
                client_updates['result'],
            )
        server.average_weights()

        logging.info('---------------------------------------------------------')
        server.log_classification_result(
            data_split='train',
            metric='f1'
        )
if __name__ == '__main__':
    run()
