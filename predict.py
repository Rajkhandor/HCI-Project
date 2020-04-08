import torch
from transformers import BertTokenizer
import torch.nn as nn
import numpy as np
from transformers import BertModel
from encoder import Encoder
from decoder import DecoderWithAttention

def initialize():
	emb_dim = 256  # dimension of word embeddings
	decoder_dim = 256  # dimension of decoder RNN
	dropout = 0.5
	decoder = DecoderWithAttention(embed_dim=emb_dim,
                                   decoder_dim=decoder_dim,
                                   vocab_size=30522,
                                   dropout=dropout)

	encoder = Encoder(256,freeze_bert = True)
	tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
	checkpoint = '../input/chatbot/checkpoint1.pt'  # path of checkpoint.pt file
	if checkpoint is not None:
	    checkpoint = torch.load(checkpoint)
	    decoder.load_state_dict(checkpoint['decoder_state_dict'])
	    encoder.load_state_dict(checkpoint['encoder_state_dict'])


def query(question,encoder,decoder,tokenizer):
	encoder = encoder.eval()
	decoder = decoder.eval()
	maxlen = 15
	tokens = tokenizer.tokenize(question) #Tokenize the sentence
	tokens = ['[CLS]'] + tokens + ['[SEP]'] #Insering the CLS and SEP token in the beginning and end of the sentence
	if len(tokens) < maxlen:
	    tokens = tokens + ['[PAD]' for _ in range(maxlen - len(tokens))] #Padding sentences
	else:
	    tokens = tokens[:maxlen-1] + ['[SEP]'] #Prunning the list to be of specified max length
	#print (tokens)
	tokens_ids = tokenizer.convert_tokens_to_ids(tokens) #Obtaining the indices of the tokens in the BERT Vocabulary
	#print (tokens_ids)
	tokens_ids_tensor = torch.tensor(tokens_ids) #Converting the list to a pytorch tensor

	#Obtaining the attention mask i.e a tensor containing 1s for no padded tokens and 0s for padded ones
	attn_mask = (tokens_ids_tensor != 0).long().unsqueeze(0)
	seq = tokens_ids_tensor.unsqueeze(0)

	# Forward prop.
	out = encoder(seq, attn_mask)
	pred = torch.LongTensor([[tokenizer.convert_tokens_to_ids('[CLS]')]])
	h,c = decoder.init_hidden_state(out) 

	sampled = []
	for t in range(20):
	    embeddings = decoder.embedding(pred).squeeze(1)  
	    attention_weighted_encoding, alpha = decoder.attention(out,h)
	    gate = decoder.sigmoid(decoder.f_beta(h))  
	    attention_weighted_encoding = gate * attention_weighted_encoding
	    h, c = decoder.decode_step(
	        torch.cat([embeddings, attention_weighted_encoding], dim=1),(h, c))  # (batch_size_t, decoder_dim)
	    pt = decoder.fc(decoder.dropout(h))  # (batch_size_t, vocab_size)
	    _,pred = pt.max(1)
	    sampled.append(pred.item())

	answer = []
	for i in sampled:
	    answer.append(tokenizer.convert_ids_to_tokens(i))
	    if (answer[-1]=='[SEP]'):
	        break

	return (answer)