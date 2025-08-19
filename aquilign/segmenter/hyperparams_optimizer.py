import json
import re
import optuna
import sys
with open(sys.argv[1], "r") as input_json:
	config_file = json.load(input_json)
if config_file["global"]["import"] != "":
	sys.path.append(config_file["global"]["import"])
import aquilign.segmenter.utils as utils
import aquilign.segmenter.models as models
import aquilign.segmenter.eval as eval
import aquilign.segmenter.datafy as datafy
import torch
import datetime
from transformers import AutoTokenizer
from torch.utils.data import DataLoader
import tqdm
from statistics import mean
import numpy as np
import os
import glob
import shutil
import sys




def objective(trial):
	hidden_size_multiplier = trial.suggest_int("hidden_size", 4, 16)
	hidden_size = hidden_size_multiplier * 8
	batch_size_multiplier = trial.suggest_int("batch_size", 2, 32)
	batch_size = batch_size_multiplier * 8
	lr = trial.suggest_float("learning_rate", 0.0001, 0.01, log=True)
	balance_class_weights = trial.suggest_categorical("balance_class_weights", [False, True])
	use_pretrained_embeddings = trial.suggest_categorical("use_pretrained_embeddings", [False, True])
	freeze_embeddings = trial.suggest_categorical("freeze_embeddings", [False, True])
	freeze_lang_embeddings = trial.suggest_categorical("freeze_lang_embeddings", [False, True])
	if use_pretrained_embeddings:
		emb_dim = 100
		add_attention_layer = False
		os.environ["TOKENIZERS_PARALLELISM"] = "false"
	else:
		emb_dim = trial.suggest_int("input_dim", 200, 300)
		add_attention_layer = trial.suggest_categorical("attention_layer", [False, True])
	# bidirectional = trial.suggest_categorical("bidirectional", [False, True])
	include_lang_metadata = trial.suggest_categorical("include_lang_metadata", [False, True])
	lang_emb_dim = trial.suggest_int("lang_emb_dim", 8, 64)

	epochs = config_file["global"]["epochs"]
	train_path = config_file["global"]["train"]
	test_path = config_file["global"]["test"]
	dev_path = config_file["global"]["dev"]
	output_dir = config_file["global"]["out_dir"]
	base_model_name = config_file["global"]["base_model_name"]
	device = "cpu"
	if device != "cpu":
		device_name = torch.cuda.get_device_name(device)
		print(f"Device name: {device_name}")

	if use_pretrained_embeddings:
		create_vocab = False
		tokenizer = AutoTokenizer.from_pretrained(base_model_name)
	else:
		create_vocab = True
		tokenizer = None
	workers = 8
	all_dataset_on_device = False
	print("Loading data")
	train_dataloader = datafy.CustomTextDataset("train",
												train_path=train_path,
												test_path=test_path,
												dev_path=dev_path,
												device=device,
												all_dataset_on_device=False,
												delimiter="£",
												output_dir=output_dir,
												create_vocab=create_vocab,
												use_pretrained_embeddings=use_pretrained_embeddings,
												debug=False,
												data_augmentation=True,
												tokenizer_name=base_model_name)
	dev_dataloader = datafy.CustomTextDataset(mode="dev",
											  train_path=train_path,
											  test_path=test_path,
											  dev_path=dev_path,
											  device=device,
											  delimiter="£",
											  output_dir=output_dir,
											  create_vocab=False,
											  input_vocab=train_dataloader.datafy.input_vocabulary,
											  lang_vocab=train_dataloader.datafy.lang_vocabulary,
											  use_pretrained_embeddings=use_pretrained_embeddings,
											  debug=False,
											  data_augmentation=True,
											  tokenizer_name=base_model_name,
											  all_dataset_on_device=False)

	loaded_dev_data = DataLoader(dev_dataloader,
									   batch_size=batch_size,
									   shuffle=False,
									   num_workers=8,
									   pin_memory=False,
									   drop_last=True)
	loaded_train_data = DataLoader(train_dataloader,
										batch_size=batch_size,
										shuffle=True,
										num_workers=workers,
										pin_memory=False,
										drop_last=True)

	output_dir = output_dir
	# On crée l'output dir:
	os.makedirs(f"{output_dir}/models/.tmp", exist_ok=True)
	os.makedirs(f"{output_dir}/best", exist_ok=True)

	print(f"Number of train examples: {len(train_dataloader.datafy.train_padded_examples)}")
	print(f"Number of test examples: {len(dev_dataloader.datafy.test_padded_examples)}")
	print(f"Total length of examples (with padding): {train_dataloader.datafy.max_length_examples}")

	input_vocab = train_dataloader.datafy.input_vocabulary
	reverse_input_vocab = {v: k for k, v in input_vocab.items()}
	lang_vocab = train_dataloader.datafy.lang_vocabulary
	target_classes = train_dataloader.datafy.target_classes
	reverse_target_classes = train_dataloader.datafy.reverse_target_classes

	corpus_size = train_dataloader.__len__()
	tgt_PAD_IDX = target_classes["[PAD]"]
	epochs = epochs

	input_dim = len(input_vocab)
	output_dim = len(target_classes)
	weights = torch.load("aquilign/segmenter/embeddings.npy")
	model = models.LSTM_Encoder(input_dim=input_dim,
									 emb_dim=emb_dim,
									 bidirectional=True,
									 lstm_dropout=0,
									 positional_embeddings=False,
									 device=device,
									 lstm_hidden_size=hidden_size,
									 batch_size=batch_size,
									 num_langs=len(lang_vocab),
									 num_lstm_layers=1,
									 include_lang_metadata=include_lang_metadata,
									 out_classes=output_dim,
									 attention=add_attention_layer,
									 lang_emb_dim=lang_emb_dim,
									 load_pretrained_embeddings=use_pretrained_embeddings,
									 pretrained_weights=weights)
	model.to(device)
	optimizer = torch.optim.Adam(model.parameters(), lr=lr)
	scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.9)

	# Les classes étant distribuées de façons déséquilibrée, on donne + d'importance à la classe <SB>
	# qu'aux deux autres pour le calcul de la loss
	if balance_class_weights:
		weights = train_dataloader.datafy.target_weights.to(device)
		criterion = torch.nn.CrossEntropyLoss(weight=weights, ignore_index=tgt_PAD_IDX)
	else:
		criterion = torch.nn.CrossEntropyLoss(ignore_index=tgt_PAD_IDX)
	print(model.__repr__())
	accuracies = []
	utils.remove_file(f"{output_dir}/accuracies.txt")
	print("Starting training")
	torch.save(input_vocab, f"{output_dir}/vocab.voc")


	# Training phase

	if use_pretrained_embeddings:
		if freeze_embeddings:
			for param in model.tok_embedding.parameters():
				param.requires_grad = False

	# Idem pour les plongements de langue. En faire un paramètre.
	if include_lang_metadata:
		if freeze_lang_embeddings:
			for param in model.lang_embedding.parameters():
				param.requires_grad = False
	results = []
	model.train()
	for epoch in range(epochs):
		epoch_number = epoch + 1
		print(f"Epoch {str(epoch_number)}")
		for examples, langs, targets in tqdm.tqdm(loaded_train_data, unit_scale=batch_size):
			# Shape [batch_size, max_length]
			# tensor_examples = examples.to(device)
			# Shape [batch_size, max_length]
			# tensor_targets = targets.to(device)
			if not all_dataset_on_device:
				examples = examples.to(device)
				targets = targets.to(device)
				langs = langs.to(device)

			# https://pytorch.org/tutorials/recipes/recipes/tuning_guide.html
			# for param in model.parameters():
			# param.grad = None
			optimizer.zero_grad()
			# Shape [batch_size, max_length, output_dim]
			output = model(examples, langs)
			# output_dim = output.shape[-1]
			# Shape [batch_size*max_length, output_dim]
			output = output.view(-1, output_dim)
			# Shape [batch_size*max_length]
			tgt = targets.view(-1)

			# output = [batch size * tgt len - 1, output dim]
			# tgt = [batch size * tgt len - 1]
			loss = criterion(output, tgt)
			loss.backward()
			torch.nn.utils.clip_grad_norm_(model.parameters(), 0.1)
			optimizer.step()

		scheduler.step()
		recall, precision, f1 = evaluate(model=model,
										 device=device,
										 loaded_dev_data=loaded_dev_data,
										 batch_size=batch_size,
										 reverse_input_vocab=reverse_input_vocab,
										 reverse_target_classes=reverse_target_classes,
										 tgt_PAD_IDX=tgt_PAD_IDX,
										 tokenizer=tokenizer)

		weighted_recall_precision = (recall[1]*2 + precision[1]) / 3
		results.append(weighted_recall_precision)
	best_result = max(results)
	return best_result

def evaluate(model,
			 device,
			 loaded_dev_data,
			 batch_size,
			 reverse_input_vocab,
			 reverse_target_classes,
			 tgt_PAD_IDX,
			 tokenizer):
	"""
	Cette fonction produit les métriques d'évaluation (justesse, précision, rappel)
	"""
	print("Evaluating model on dev data")
	debug = False
	epoch_accuracy = []
	epoch_loss = []
	# Timer = utils.Timer()
	all_preds = []
	all_targets = []
	all_examples = []
	model.eval()
	for examples, langs, targets in tqdm.tqdm(loaded_dev_data, unit_scale=batch_size):
		# https://discuss.pytorch.org/t/should-we-set-non-blocking-to-true/38234/3
		# Timer.start_timer("preds")
		tensor_examples = examples.to(device)
		tensor_langs = langs.to(device)
		tensor_target = targets.to(device)
		with torch.no_grad():
			# On prédit. La langue est toujours envoyée même si elle n'est pas traitée par le modèle, pour des raisons de simplicité
			preds = model(tensor_examples, tensor_langs)
			all_preds.append(preds)
			all_targets.append(targets)
			all_examples.append(examples)

	# On crée une seul vecteur, en concaténant tous les exemples sur la dimension 0 (= chaque exemple individuel)
	cat_preds = torch.cat(all_preds, dim=0)
	cat_targets = torch.cat(all_targets, dim=0)
	cat_examples = torch.cat(all_examples, dim=0)
	results = eval.compute_metrics(predictions=cat_preds,
								   labels=cat_targets,
								   examples=cat_examples,
								   id_to_word=reverse_input_vocab,
								   idx_to_class=reverse_target_classes,
								   padding_idx=tgt_PAD_IDX,
								   batch_size=batch_size,
								   last_epoch=False,
								   tokenizer=tokenizer)

	recall = ["Recall", results["recall"][0], results["recall"][1]]
	precision = ["Precision", results["precision"][0], results["precision"][1]]
	f1 = ["F1", results["f1"][0], results["f1"][1]]
	return recall, precision, f1



if __name__ == '__main__':
	study = optuna.create_study(direction='maximize')
	study.optimize(objective, n_trials=100)
	print("Best Hyperparameters:", study.best_params)