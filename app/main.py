# -*- coding: utf-8 -*-

from datetime import datetime
from utils import accuracy_score, create_ts_dir_res, get_dataloaders, get_datasets
from transformers import BertTokenizer, BertModel

from approaches.MainApproch import MainApproch
from approaches.LayerWise import LayerWise
from approaches.LayerAggregation import LayerAggregation

from competitros.BertLinears import BertLinears
from competitros.BertLSTM import BertLSTM
from competitros.BertGRU import BertGRU

from Baselines import Baselines

import torch
import torch.nn as nn

import copy


device = torch.device('cuda:2' if torch.cuda.is_available() else 'cpu')


def main():
    
	embedding_split_perc = 0.1
    
	batch_size = 64
	epochs = 5
	patience = 3

	tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
	model = BertModel.from_pretrained("bert-base-uncased").to(device)
  
	dataloaders = get_dataloaders(get_datasets(), tokenizer, batch_size)
	datasets_name = list(dataloaders.keys())

	loss_fn = nn.CrossEntropyLoss()
 
	timestamp = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
	create_ts_dir_res(timestamp)
 
	params = {
		'device': device,
		'batch_size': batch_size,
  		'model': model,
    	'tokenizer': tokenizer,
     	'embedding_split_perc': embedding_split_perc,
      	'loss_fn': loss_fn,
		'score_fn': accuracy_score,
		'patience': patience,
		'epochs': epochs,
	}

	# our approaches
	main_approach = MainApproch(device, dataloaders, model, tokenizer, embedding_split_perc, timestamp)
	layer_wise = LayerWise(device, dataloaders, model, tokenizer, embedding_split_perc, timestamp)
	layer_aggregation = LayerAggregation(copy.deepcopy(params), dataloaders, timestamp)
 
 
	our_approaces_names = [main_approach.__class__.__name__, layer_wise.__class__.__name__, layer_aggregation.__class__.__name__, ]
 
	
	# competitors
	bert_linears = BertLinears(copy.deepcopy(params), dataloaders, timestamp)
	bert_lstm = BertLSTM(copy.deepcopy(params), dataloaders, timestamp, bidirectional=False)
	bert_lstm_bi = BertLSTM(copy.deepcopy(params), dataloaders, timestamp, bidirectional=True)
	bert_gru = BertGRU(copy.deepcopy(params), dataloaders, timestamp)
 
	# baselines
	baselines = Baselines(datasets_name, timestamp, our_approaces_names)
 
 
 
	methods = [
		# our approaches
		main_approach, layer_wise, layer_aggregation,

		# competitors
		#bert_linears, bert_lstm, bert_lstm_bi, bert_gru,
  
		# baselines
		baselines
	]
 
	for method in methods:
		method.run()


if __name__ == "__main__":
    main()
    

