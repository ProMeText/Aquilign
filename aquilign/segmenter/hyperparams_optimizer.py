import argparse
import datetime
import decimal
import json
import optuna
import sys
from functools import partial

parser = argparse.ArgumentParser()
parser.add_argument("-a", "--architecture", default="lstm",
					help="Architecture to be tested")
parser.add_argument("-p", "--parameters", default=None,
					help="Path to parameters file")
parser.add_argument("-size", "--model_size", default=False,
					help="Include parameters numbers in optimization")
parser.add_argument("-d", "--debug", default=False,
					help="Debug mode")
parser.add_argument("-t", "--trials", default=50,
					help="Number of trials")
parser.add_argument("-e", "--epochs", default=10,
					help="Numbr of epochs per trial")
parser.add_argument("-pt", "--pretrained", default=True,
					help="Use pretrained embeddings")
parser.add_argument("-bt", "--bert_tokenizer", default=False,
					help="Use bert tokenizer (without pre-trained embeddings)")
parser.add_argument("-n", "--out_name", default="",
					help="Prefix to add to output dir")
args = parser.parse_args()
architecture = args.architecture
debug = args.debug
model_size = args.model_size
parameters = args.parameters
use_pretrained_embeddings = True if args.pretrained == "True" else False
use_bert_tokenizer = True if args.bert_tokenizer == "True" else False
out_name = args.out_name
if out_name != "":
	out_name = f"_{out_name}"
trials = int(args.trials)

with open(parameters, "r") as input_json:
	config_file = json.load(input_json)
if config_file["global"]["import"] != "":
	sys.path.append(config_file["global"]["import"])
import aquilign.segmenter.utils as utils
import aquilign.segmenter.models as models
import aquilign.segmenter.eval as eval
import aquilign.segmenter.datafy as datafy
import torch
from transformers import AutoTokenizer
from torch.utils.data import DataLoader
import tqdm
import os



def objective(trial, bert_train_dataloader, bert_dev_dataloader, no_bert_train_dataloader, no_bert_dev_dataloader, architecture, model_size, use_pretrained_embeddings, use_bert_tokenizer):
	os.environ["TOKENIZERS_PARALLELISM"] = "false"
	base_model_name = config_file["global"]["base_model_name"]
	balance_class_weights = trial.suggest_categorical("balance_class_weights", [False, True])
	if architecture == "BERT":
		lr = trial.suggest_float("learning_rate", 0.00004, 0.00005, log=False)
		batch_size = 32
		base_model_name = "google-bert/bert-base-multilingual-cased"
	elif architecture == "DISTILBERT":
		lr = trial.suggest_float("learning_rate", 0.00004, 0.00005, log=False)
		batch_size = 32
		base_model_name = "distilbert/distilbert-base-multilingual-cased"
	else:
		lr = trial.suggest_float("learning_rate", 0.0001, 0.01, log=True)
		hidden_size = trial.suggest_int("hidden_size", 8, 160, step=8)
		linear_layers = trial.suggest_int("linear_layers", 1, 4)
		linear_layers_hidden_size = trial.suggest_categorical("linear_layers_hidden_size", [32, 64, 128, 256])
		if architecture == "lstm":
			num_lstm_layers = trial.suggest_int("num_lstm_layers", 1, 4)
			linear_dropout = trial.suggest_float("linear_dropout", 0.0, 0.5)
			if num_lstm_layers == 1:
				lstm_dropout = 0
			else:
				lstm_dropout = trial.suggest_float("lstm_dropout", 0, 0.8)
		elif architecture == "gru":
			num_gru_layers = trial.suggest_int("num_gru_layers", 1, 2)
			gru_dropout = trial.suggest_float("gru_dropout", 0, 0.5)
		elif architecture == "cnn":
			num_cnn_layers = trial.suggest_int("num_cnn_layers", 1, 15)
			positional_embeddings = trial.suggest_categorical("positional_embeddings", [False, True])
			kernel_size = trial.suggest_int("kernel_size", 1, 15, step=2)
			cnn_dropout = trial.suggest_float("cnn_dropout", 0, 0.8)
			cnn_scale = trial.suggest_float("cnn_scale", 0, 0.8)
			linear_dropout = trial.suggest_float("linear_dropout", 0.0, 0.5)

		if architecture not in  ["transformers", "BERT", "DISTILBERT"]:
			add_attention_layer = trial.suggest_categorical("attention_layer", [False, True])
			batch_size = trial.suggest_int("batch_size", 16, 128, step=16)
		else:
			if architecture == "transformers":
				num_transformers_layers = trial.suggest_int("num_transformers_layers", 1, 4)
			batch_size = trial.suggest_int("batch_size", 16, 64, step=16)
	if architecture not in ["BERT", "DISTILBERT"]:
		if use_pretrained_embeddings is True:
			train_dataloader = bert_train_dataloader
			dev_dataloader = bert_dev_dataloader
			emb_dim = 100
			use_bert_tokenizer = True
			keep_bert_dimensions = True
		else:
			keep_bert_dimensions = False
			emb_dim = trial.suggest_int("input_dim", 300, 400, step=8)
			if use_bert_tokenizer is True:
				print("Using Bert tokenized data")
				train_dataloader = bert_train_dataloader
				dev_dataloader = bert_dev_dataloader
			else:
				print("Using homemade tokenization")
				train_dataloader = no_bert_train_dataloader
				dev_dataloader = no_bert_dev_dataloader
		if use_bert_tokenizer and not use_pretrained_embeddings:
			# pass
			# freeze_embeddings = False
			freeze_embeddings = trial.suggest_categorical("freeze_embeddings", [False, True])
		else:
			freeze_embeddings = trial.suggest_categorical("freeze_embeddings", [False, True])
		include_lang_metadata = trial.suggest_categorical("include_lang_metadata", [False, True])
		if include_lang_metadata:
			freeze_lang_embeddings = trial.suggest_categorical("freeze_lang_embeddings", [False, True])
			lang_emb_dim = trial.suggest_int("lang_emb_dim", 8, 64, step=8)
		else:
			freeze_lang_embeddings = False
			lang_emb_dim = 4
	else:
		train_dataloader = bert_train_dataloader
		dev_dataloader = bert_dev_dataloader
		use_pretrained_embeddings = False
		include_lang_metadata = False
		use_bert_tokenizer = False


	epochs = config_file["global"]["epochs"]
	device = config_file["global"]["device"]
	output_dir = config_file["global"]["out_dir"]
	if device != "cpu":
		device_name = torch.cuda.get_device_name(device)
		print(f"Device name: {device_name}")
	if architecture in ["BERT", "DISTILBERT"] or use_pretrained_embeddings:
		tokenizer = AutoTokenizer.from_pretrained(base_model_name)
	else:
		tokenizer = None
	workers = 8
	params = trial.params
	print(f"Current params: {params}")
	print("Loading data")



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

	tgt_PAD_IDX = target_classes["[PAD]"]
	epochs = epochs

	input_dim = len(input_vocab)
	output_dim = len(target_classes)
	weights = torch.load("aquilign/segmenter/embeddings.npy")
	if architecture == "lstm":
		model = models.LSTM_Encoder(input_dim=input_dim,
										 emb_dim=emb_dim,
										 bidirectional=True,
										 lstm_dropout=lstm_dropout,
										 positional_embeddings=False,
										 device=device,
										 lstm_hidden_size=hidden_size,
										 batch_size=batch_size,
										 num_langs=len(lang_vocab),
										 num_lstm_layers=num_lstm_layers,
										 include_lang_metadata=include_lang_metadata,
										 out_classes=output_dim,
										 attention=add_attention_layer,
										 lang_emb_dim=lang_emb_dim,
										 load_pretrained_embeddings=use_pretrained_embeddings,
										 pretrained_weights=weights,
										 linear_layers=linear_layers,
										 linear_layers_hidden_size=linear_layers_hidden_size,
										 use_bert_tokenizer=use_bert_tokenizer,
										 keep_bert_dimensions=keep_bert_dimensions,
										linear_dropout=linear_dropout)
	elif architecture == "transformers":
		model = models.TransformerModel(input_dim=input_dim,
											 emb_dim=emb_dim,
											 num_heads=8,
											 num_layers=num_transformers_layers,
											 device=device,
											 output_dim=output_dim,
											 num_langs=len(lang_vocab),
											 lang_emb_dim=lang_emb_dim,
											 include_lang_metadata=include_lang_metadata,
										     linear_layers=linear_layers,
											 linear_layers_hidden_size=linear_layers_hidden_size,
										     load_pretrained_embeddings=use_pretrained_embeddings,
										 	  use_bert_tokenizer=use_bert_tokenizer,
										      pretrained_weights=weights)
	elif architecture == "gru":
		model = models.GRU_Encoder(input_dim=input_dim,
								   emb_dim=emb_dim,
								   bidirectional=True,
								   dropout=gru_dropout,
								   positional_embeddings=False,
								   device=device,
								   hidden_size=hidden_size,
								   batch_size=batch_size,
								   num_langs=len(lang_vocab),
								   num_layers=num_gru_layers,
								   include_lang_metadata=include_lang_metadata,
								   out_classes=output_dim,
								   attention=add_attention_layer,
								   lang_emb_dim=lang_emb_dim,
								   load_pretrained_embeddings=use_pretrained_embeddings,
								   pretrained_weights=weights
								   )
	elif architecture == "cnn":
		model = models.CnnEncoder(input_dim=input_dim,
								  emb_dim=emb_dim,
								  dropout=cnn_dropout,
								  kernel_size=kernel_size,
								  positional_embeddings=positional_embeddings,
								  device=device,
								  hidden_size=hidden_size,
								  num_langs=len(lang_vocab),
								  num_conv_layers=num_cnn_layers,
								  include_lang_metadata=include_lang_metadata,
								  out_classes=output_dim,
								  attention=add_attention_layer,
								  lang_emb_dim=lang_emb_dim,
								  load_pretrained_embeddings=use_pretrained_embeddings,
								  use_bert_tokenizer=use_bert_tokenizer,
								  linear_layers_hidden_size=linear_layers_hidden_size,
								  linear_layers=linear_layers,
								  pretrained_weights=weights,
								  cnn_scale=cnn_scale,
								  keep_bert_dimensions=keep_bert_dimensions,
								  linear_dropout=linear_dropout
								  )

	elif architecture == "Baseline":
		model = models.BaseLineModel(input_dim=input_dim,
										 emb_dim=emb_dim,
										 positional_embeddings=False,
										 device=device,
										 batch_size=batch_size,
										 num_langs=len(lang_vocab),
										 include_lang_metadata=include_lang_metadata,
										 out_classes=output_dim,
										 attention=add_attention_layer,
										 lang_emb_dim=lang_emb_dim,
										 load_pretrained_embeddings=use_pretrained_embeddings,
										 pretrained_weights=weights,
										 linear_layers=linear_layers,
										 linear_layers_hidden_size=linear_layers_hidden_size,
										 use_bert_tokenizer=use_bert_tokenizer,
										 keep_bert_dimensions=keep_bert_dimensions,
										linear_dropout=linear_dropout)
	elif architecture == "BERT":
		from transformers import AutoModelForTokenClassification
		model = AutoModelForTokenClassification.from_pretrained(base_model_name, num_labels=3)
	elif architecture == "DISTILBERT":
		from transformers import DistilBertForTokenClassification
		model = DistilBertForTokenClassification.from_pretrained(base_model_name, num_labels=3)



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


	# Training phase

	if use_pretrained_embeddings:
		if freeze_embeddings:
			for param in model.embedding.parameters():
				param.requires_grad = False

	# Idem pour les plongements de langue. En faire un paramètre.
	if include_lang_metadata:
		if freeze_lang_embeddings:
			for param in model.lang_embedding.parameters():
				param.requires_grad = False
	results = []
	params_number = sum(p.numel() for p in model.parameters())
	params_number_to_print = format(decimal.Decimal(params_number), '.4e')
	for epoch in range(epochs):
		model.train()
		epoch_number = epoch + 1
		print(f"Epoch {str(epoch_number)}")
		for data in tqdm.tqdm(loaded_train_data, unit_scale=batch_size):
			if architecture in ["BERT", "DISTILBERT"]:
				examples, masks, targets = data
				masks = masks.to(device)
			else:
				examples, langs, targets = data
				langs = langs.to(device)
			examples = examples.to(device)
			targets = targets.to(device)
			# Shape [batch_size, max_length]
			# tensor_examples = examples.to(device)
			# Shape [batch_size, max_length]
			# tensor_targets = targets.to(device)

			# https://pytorch.org/tutorials/recipes/recipes/tuning_guide.html
			# for param in model.parameters():
			# param.grad = None
			optimizer.zero_grad()
			# Shape [batch_size, max_length, output_dim]
			if architecture not in  ["BERT", "DISTILBERT"]:
				output = model(examples, langs)
				output = output.view(-1, output_dim)
				tgt = targets.view(-1)
				loss = criterion(output, tgt)
			else:
				output = model(input_ids=examples, attention_mask=masks, labels=targets)
				loss = output.loss
			# output_dim = output.shape[-1]
			# Shape [batch_size*max_length, output_dim]
			# Shape [batch_size*max_length]

			# output = [batch size * tgt len - 1, output dim]
			# tgt = [batch size * tgt len - 1]

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
										 tokenizer=tokenizer,
										 architecture=architecture)

		weighted_recall_precision = (recall[2]*1.3 + precision[2]) / 2.3
		f1_score = f1[2]
		# results.append(weighted_recall_precision)
		results.append(f1_score)
		with open(f"../trash/segmenter_hyperparasearch_{architecture}_{date_hour}{out_name}.txt", "a") as f:
			f.write(f"Epoch {epoch_number}: weighted: {round(weighted_recall_precision, 4)}, F1: {round(f1[2], 4)} (recall: {round(recall[2], 4)}, precision: {round(precision[2], 4)})\n")
			if epoch_number == epochs:
				f.write(f"Nombre de paramètres: {params_number_to_print}\n")
	best_result = max(results)
	print(f"Best epoch result: {best_result}")
	if model_size:
		return best_result, params_number
	else:
		return best_result

def evaluate(model,
			 device,
			 loaded_dev_data,
			 batch_size,
			 reverse_input_vocab,
			 reverse_target_classes,
			 tgt_PAD_IDX,
			 tokenizer,
			 architecture):
	"""
	Cette fonction produit les métriques d'évaluation (justesse, précision, rappel)
	"""
	print("Evaluating model on dev data")
	debug = False
	all_preds = []
	all_targets = []
	all_examples = []
	model.eval()
	for data in tqdm.tqdm(loaded_dev_data, unit_scale=batch_size):
		if architecture in ["BERT", "DISTILBERT"]:
			examples, masks, targets = data
			masks = masks.to(device)
		else:
			examples, langs, targets = data
			langs = langs.to(device)
		examples = examples.to(device)
		targets = targets.to(device)

		# https://discuss.pytorch.org/t/should-we-set-non-blocking-to-true/38234/3
		# Timer.start_timer("preds")
		with torch.no_grad():
			# On prédit. La langue est toujours envoyée même si elle n'est pas traitée par le modèle, pour des raisons de simplicité
			if architecture not in ["BERT", "DISTILBERT"]:
				preds = model(examples, langs)
			else:
				preds = model(input_ids=examples, attention_mask=masks, labels=targets).logits
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


def print_trial_info(study, trial):
	with open(f"../trash/segmenter_hyperparasearch_{architecture}_{date_hour}{out_name}.txt", "a") as f:
		f.write(f"Trial {trial.number} - Paramètres : {trial.params}\n")
		if not model_size:
			f.write(f"Valeur de la métrique : {trial.value}\n")
		f.write(f"---\n")

if __name__ == '__main__':

	date_hour = datetime.datetime.now().isoformat()
	if os.path.exists(f"../trash/segmenter_hyperparasearch_{architecture}_{date_hour}{out_name}.txt"):
		os.remove(f"../trash/segmenter_hyperparasearch_{architecture}_{date_hour}{out_name}.txt")
	train_path = config_file["global"]["train"]
	test_path = config_file["global"]["test"]
	device = config_file["global"]["device"]
	dev_path = config_file["global"]["dev"]
	output_dir = config_file["global"]["out_dir"] + f"/{date_hour}"
	base_model_name = config_file["global"]["base_model_name"]
	data_augmentation = config_file["global"]["data_augmentation"]
	pretrained_train_dataloader = datafy.CustomTextDataset("train",
												train_path=train_path,
												test_path=test_path,
												dev_path=dev_path,
												device=device,
												delimiter="£",
												output_dir=output_dir,
												create_vocab=False,
												use_pretrained_embeddings=True,
												debug=debug,
												data_augmentation=data_augmentation,
												tokenizer_name=base_model_name,
												architecture=architecture,
														   tuning_mode=True)
	pretrained_dev_dataloader = datafy.CustomTextDataset(mode="dev",
											  train_path=train_path,
											  test_path=test_path,
											  dev_path=dev_path,
											  device=device,
											  delimiter="£",
											  output_dir=output_dir,
											  create_vocab=False,
											  input_vocab=pretrained_train_dataloader.datafy.input_vocabulary,
											  lang_vocab=pretrained_train_dataloader.datafy.lang_vocabulary,
											  use_pretrained_embeddings=True,
											  debug=debug,
											  data_augmentation=data_augmentation,
											  tokenizer_name=base_model_name,
												architecture=architecture,
														   tuning_mode=True)


	not_pretrained_train_dataloader = datafy.CustomTextDataset("train",
												train_path=train_path,
												test_path=test_path,
												dev_path=dev_path,
												device=device,
												delimiter="£",
												output_dir=output_dir,
												create_vocab=True,
												use_pretrained_embeddings=False,
												debug=debug,
												data_augmentation=data_augmentation,
												tokenizer_name=base_model_name,
												architecture=architecture,
														   tuning_mode=True)
	not_pretrained_dev_dataloader = datafy.CustomTextDataset(mode="dev",
											  train_path=train_path,
											  test_path=test_path,
											  dev_path=dev_path,
											  device=device,
											  delimiter="£",
											  output_dir=output_dir,
											  create_vocab=False,
											  input_vocab=not_pretrained_train_dataloader.datafy.input_vocabulary,
											  lang_vocab=not_pretrained_train_dataloader.datafy.lang_vocabulary,
											  use_pretrained_embeddings=False,
											  debug=debug,
											  data_augmentation=data_augmentation,
											  tokenizer_name=base_model_name,
												architecture=architecture,
														   tuning_mode=True)

	if model_size:
		study = optuna.create_study(directions=['maximize', 'minimize'])
	else:
		study = optuna.create_study(direction='maximize')
	objective = partial(objective,
						bert_train_dataloader=pretrained_train_dataloader,
						bert_dev_dataloader=pretrained_dev_dataloader,
						no_bert_train_dataloader=not_pretrained_train_dataloader,
						no_bert_dev_dataloader=not_pretrained_dev_dataloader,
						architecture=architecture,
						model_size=model_size,
						use_pretrained_embeddings=use_pretrained_embeddings,
						use_bert_tokenizer=use_bert_tokenizer)
	study.optimize(objective, n_trials=trials, callbacks=[print_trial_info])
	with open(f"../trash/segmenter_hyperparasearch_{architecture}_{date_hour}{out_name}.txt", "a") as f:
		f.write((str(study.best_trial) + "\n"))
		if model_size:
			f.write((str(study.best_trials) + "\n"))
		else:
			f.write((str(study.best_trial) + "\n"))
		f.write(str(study.best_params))
	print("Best Hyperparameters:", study.best_params)