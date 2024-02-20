# -*- coding: utf-8 -*-

from datetime import datetime
import argparse

from BaseEmbedding import BaseEmbedding
from utils import accuracy_score, create_ts_dir_res, get_datasets
from TextDataset import get_dsname_dataloaders

from Approaches import MainApproch, LayerWise, LayerAggregation
from Competitors import Linear, LSTMGRU
from Baselines import Baselines

import torch
import torch.nn as nn
from transformers import DistilBertTokenizer, DistilBertModel, BertTokenizer, BertModel


parser = argparse.ArgumentParser()
parser.add_argument('-s', '--strategies', type=str, choices=['our_approaches', 'competitors', 'baselines'], nargs='+', required=True, help='Possible strategies to run')
parser.add_argument('-a', '--ablations', type=bool, required=True, help='Bool ablations')
parser.add_argument('-m', '--model', type=str, choices=['BERT', 'DISTILBERT'], required=True, help='Pretreined BERT model from Huggingface')

args = parser.parse_args()

base_embeds_model = args.model
bool_ablations = args.ablations


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


bert_models = {
	'BERT': {
		'model': BertModel.from_pretrained("bert-base-uncased"),
		'tokenizer': BertTokenizer.from_pretrained('bert-base-uncased'),
		'n_layers': 12
	},
	'DISTILBERT': {
		'model': DistilBertModel.from_pretrained("distilbert-base-uncased"),
		'tokenizer': DistilBertTokenizer.from_pretrained('distilbert-base-uncased'),
		'n_layers': 6
	}
}


def run_methods(methods):
    
	for methods_group, methods_list in methods.items():
		print(f'---------------------------------- RUNNING {methods_group} ----------------------------------')
		for method in methods_list: method.run()
 
	if bool_ablations:
		# run ablations
		for ablation in methods['our_approches']:
			ablation.bool_ablations = True
			ablation.run()
  
 
def main():
    	
	batch_size = 128
	epochs = 100
	patience = 20

	tokenizer = bert_models[base_embeds_model]['tokenizer']
	model = bert_models[base_embeds_model]['model']
	model_hidden_size = model.config.hidden_size
 
	print('=> Getting data')
	ds_name_dataloaders = get_dsname_dataloaders(get_datasets(), tokenizer, batch_size)
	datasets_name = list(ds_name_dataloaders.keys())
	print(' DONE\n')
 
	timestamp = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
	create_ts_dir_res(timestamp)
 
	print(f'=> Obtaining Pretrained {base_embeds_model} Embeddings')
	be = BaseEmbedding(model, device, ds_name_dataloaders, bert_models[base_embeds_model]['n_layers'])
	be.save_base_embeddings(base_embeds_model)
	print(' DONE\n')
 
	training_params = {
		'device': device,
		'batch_size': batch_size,
  		'model': model,
	  	'loss_fn': nn.CrossEntropyLoss(),
		'score_fn': accuracy_score,
		'patience': patience,
		'epochs': epochs,
	}
 
	common_parmas = {
		'datasets_name': datasets_name, 
		'timestamp': timestamp,
		'base_embeds_model': base_embeds_model
	}

 
	methods = {
		# our approaches
		'our_approches': [
			MainApproch(common_parmas, False), 
			LayerWise(common_parmas, bert_models[base_embeds_model]['n_layers'] * 768, False),
			LayerWise(common_parmas, 768, False),
			LayerAggregation(training_params, common_parmas, bert_models[base_embeds_model]['n_layers'], 768, False)
		],

		# competitors
		'competitors': [
			Linear(training_params, common_parmas, model_hidden_size),
			LSTMGRU(training_params, common_parmas, model_hidden_size, 'LSTM', bidirectional=False),
			LSTMGRU(training_params, common_parmas, model_hidden_size, 'LSTM', bidirectional=True),
			LSTMGRU(training_params, common_parmas, model_hidden_size, 'GRU', bidirectional=False),
			LSTMGRU(training_params, common_parmas, model_hidden_size, 'GRU', bidirectional=True)
		],
  
		# baselines
		'baselines': [ Baselines(common_parmas) ] 
	}
 
	run_methods(methods)

if __name__ == "__main__":
	print(f'Running Application on {device}\n')
	main()
	

