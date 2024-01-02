# -*- coding: utf-8 -*-

from utils import accuracy_score, get_dataloaders, get_datasets
from transformers import BertTokenizer, BertModel

from approaches.MainApproch import MainApproch
from approaches.LayerWise import LayerWise
from approaches.LayerAggregation import LayerAggregation

from competitros.BertLinears import BertLinears
from competitros.BertLSTM import BertLSTM
from competitros.BertGRU import BertGRU

import torch
import torch.nn as nn


device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')


def main():
    
	embedding_split_perc = 0.1
    
	batch_size = 64
	epochs = 10
	patience = 3

	tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
	model = BertModel.from_pretrained("bert-base-uncased").to(device)
  
	dataloaders = get_dataloaders(get_datasets(), tokenizer, batch_size)

	loss_fn = nn.CrossEntropyLoss()
 
	params = {
		'device': device,
		'batch_size': batch_size,
  		'model': model,
    	'tokenizer': tokenizer,
     	'embedding_split_perc': embedding_split_perc,
      	'loss_fn': loss_fn,
		'score_fn': accuracy_score,
		'patience': patience,
		'epochs': epochs
	}

	# our approaches
	main_approach = MainApproch(device, dataloaders, model, tokenizer, embedding_split_perc)
	layer_wise = LayerWise(device, dataloaders, model, tokenizer, embedding_split_perc)
	layer_aggregation = LayerAggregation(params)
 
	
	# competitors
	bert_linears = BertLinears(params, dataloaders)
 
	bert_lstm = BertLSTM(params, dataloaders, bidirectional=False)
 
	bert_lstm_bi = BertLSTM(params, dataloaders, bidirectional=True)
 
	bert_gru = BertGRU(params)
 
 
 
	methods = [
		# our approaches
		#main_approach,
		layer_wise,
		layer_aggregation,

		# competitors
		#bert_linears,
		#bert_lstm,
		#bert_lstm_bi,
		#bert_gru
  
		# baselines
	]
 
	for method in methods:
		method.run()


if __name__ == "__main__":
    main()
    

