import collections
import numpy as np
import pandas as pd
import copy, pdb, time, warnings, torch


from torch import nn
import torch.nn.functional as F
from torch.utils import data
from sklearn.metrics import confusion_matrix
from torch.utils.data import DataLoader, Dataset
from sklearn.metrics import accuracy_score, recall_score

# import optimizer
from fed_multimodal_restcol.trainer.optimizer import FedProxOptimizer
from fed_multimodal_restcol.trainer.evaluation import EvalMetric


class ClientFedAvg(object):
    def __init__(
        self, 
        args, 
        device, 
        criterion, 
        dataloader, 
        model, 
        label_dict=None,
        num_class=None
    ):
        self.args = args
        self.model = model
        self.device = device
        self.criterion = criterion
        self.dataloader = dataloader
        self.multilabel = True if args.dataset == 'ptb-xl' else False
        
    def get_parameters(self):
        # Return model parameters
        return self.model.state_dict()
    
    def get_model_result(self):
        # Return model results
        return self.result
    
    def get_test_true(self):
        # Return test labels
        return self.test_true
    
    def get_test_pred(self):
        # Return test predictions
        return self.test_pred
    
    def get_train_groundtruth(self):
        # Return groundtruth used for training
        return self.train_groundtruth

    def update_weights(self):
        # Set mode to train model
        self.model.train()

        # initialize eval
        self.eval = EvalMetric(self.multilabel)
        
        # optimizer
        if self.args.fed_alg in ['fed_avg', 'fed_opt']:
            optimizer = torch.optim.SGD(
                self.model.parameters(), 
                lr=self.args.learning_rate,
                momentum=0.9,
                weight_decay=1e-5
            )
        else:
            optimizer = FedProxOptimizer(
                self.model.parameters(), 
                lr=self.args.learning_rate,
                momentum=0.9,
                weight_decay=1e-5,
                mu=self.args.mu
            )
            
        # last global model
        last_global_model = copy.deepcopy(self.model)
        
        for iter in range(int(self.args.local_epochs)):
            for batch_idx, batch_data in enumerate(self.dataloader):
                self.model.zero_grad()
                optimizer.zero_grad()

                if self.args.modality == "multimodal":
                    x_a, x_b, l_a, l_b, y = batch_data
                    x_a, x_b, y = x_a.to(self.device), x_b.to(self.device), y.to(self.device)
                    #print("[DEBUG] x_a:", x_a.mean().item(), x_a.std().item())
                    #print("[DEBUG] x_b:", x_b.mean().item(), x_b.std().item())
                    #print("[DEBUG] y:", y.view(-1).tolist()[:5])

                    preds, _ = self.model(x_a.float(), x_b.float())

                else:
                    x, l, y = batch_data
                    x, y = x.to(self.device), y.to(self.device)
                    #print("[DEBUG] x:", x.mean().item(), x.std().item())
                    #print("[DEBUG] y:", y.view(-1).tolist()[:5])

                    preds, _ = self.model(x.float(), l)

                y = y.view(-1,1).float()
                loss = self.criterion(preds, y)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 10.0)
                optimizer.step()


        self.result = {
            "loss": float(loss.item()),
            "sample": len(self.dataloader.dataset),
            "rmse": float(torch.sqrt(F.mse_loss(preds, y)).item()),
            "mae": float(F.l1_loss(preds, y).item())
            }
        print("[DEBUG] Final result dict:", self.result)

