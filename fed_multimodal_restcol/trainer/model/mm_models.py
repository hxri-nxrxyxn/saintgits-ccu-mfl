import pdb
import torch
import numpy as np
import torch.nn as nn

from torch import Tensor
from torch.nn import functional as F
from torch.nn.utils.rnn import pad_packed_sequence, pack_padded_sequence

from typing import Dict, Iterable, Optional

class ImageTextClassifier(nn.Module):
    def __init__(
        self,
        num_classes: int,        # Number of output classes (or 1 for regression)
        img_input_dim: int,      # Input dimension of image features
        text_input_dim: int,     # Input dimension of text features
        d_hid: int = 64,          # Hidden dimension used in projection layers
        en_att: bool = False,
        att_name: str='',       # Attention Name
        d_head: int=6           # Head dim
    ):
        super(ImageTextClassifier, self).__init__()
        self.dropout_p = 0.1     # Dropout probability

        # Projection MLP for image features
        self.img_proj = nn.Sequential(
            nn.Linear(img_input_dim, d_hid),
            nn.ReLU(),
            nn.Dropout(self.dropout_p),
            nn.Linear(d_hid, d_hid)
        )

        # Projection MLP for text features
        self.text_proj = nn.Sequential(
            nn.Linear(text_input_dim, d_hid),
            nn.ReLU(),
            nn.Dropout(self.dropout_p),
            nn.Linear(d_hid, d_hid)
        )

        # Final classification head (takes concatenated image + text features)
        self.classifier = nn.Sequential(
            nn.Linear(d_hid * 2, 64),
            nn.ReLU(),
            nn.Dropout(self.dropout_p),
            #nn.Linear(64, num_classes)
            nn.Linear(64, 1)
        )

        self.init_weight()

    def init_weight(self):
        # Apply Xavier initialization to all linear layers
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    m.bias.data.fill_(0.01)

    def forward(self, x_img, x_text):
        # Project image features 
        x_img = self.img_proj(x_img.mean(dim=1))     # shape: [B, d_hid]
        #print("[DEBUG] Using ImageTextClassifier from:", __file__)
        # Project text features
        if x_text.dim() == 3:
        	x_text = x_text.mean(dim=1)
        x_text = self.text_proj(x_text)

        # Concatenate both modalities
        x_mm = torch.cat((x_img, x_text), dim=1)  # shape: [B, d_hid*2]
        
        

        # Run through classifier
        preds = self.classifier(x_mm)
        return preds, x_mm   # return both predictions and the fused features
