import torch
from transformers import BertTokenizer
import torch.nn as nn
import numpy as np
from transformers import BertModel

class Encoder(nn.Module):

    def __init__(self, encode_dim, freeze_bert = True):
        super(Encoder, self).__init__()
        #Instantiating BERT model object 
        self.bert_layer = BertModel.from_pretrained('bert-base-uncased')
        self.encode_dim = encode_dim
        #Freeze bert layers
        if freeze_bert:
            for p in self.bert_layer.parameters():
                p.requires_grad = False
        
        self.fc = nn.Linear(768,encode_dim)
        

    def forward(self, seq, attn_masks):
        '''
        Inputs:
            -seq : Tensor of shape [B, T] containing token ids of sequences
            -attn_masks : Tensor of shape [B, T] containing attention masks to be used to avoid contibution of PAD tokens
        '''

        #Feeding the input to BERT model to obtain contextualized representations
        cont_reps, _ = self.bert_layer(seq, attention_mask = attn_masks)
        #print (cont_reps.size())  (batch_size , max_len, 768) 
        self.out = self.fc(cont_reps) # (batch_size, max_len, encode_dim)
        #print (self.out.size())
        return self.out