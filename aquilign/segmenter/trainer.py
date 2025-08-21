import re
from platform import architecture

from transformers import AutoTokenizer

import aquilign.segmenter.utils as utils
import aquilign.segmenter.models as models
import aquilign.segmenter.eval as eval
import aquilign.segmenter.datafy as datafy
import torch
import datetime
from torch.utils.data import DataLoader
import tqdm
from statistics import mean
import numpy as np
import os
import glob
import shutil
import sys
class Trainer:
	def  __init__(self,
				  config_file):

		architecture = sys.argv[2]
		if len(sys.argv) == 4:
			self.debug = True if sys.argv[3] == "True" else False
		else:
			self.debug = False

		fine_tune = False
		epochs = config_file["global"]["epochs"]
		batch_size = config_file["global"]["batch_size"]
		lr = config_file["global"]["lr"]
		device = config_file["global"]["device"]
		workers = config_file["global"]["workers"]
		train_path = config_file["global"]["train"]
		test_path = config_file["global"]["test"]
		dev_path = config_file["global"]["dev"]
		output_dir = config_file["global"]["out_dir"]
		base_model_name = config_file["global"]["base_model_name"]
		use_pretrained_embeddings = config_file["global"]["use_pretrained_embeddings"]
		use_bert_tokenizer = config_file["global"]["use_bert_tokenizer"]
		if use_pretrained_embeddings or use_bert_tokenizer:
			os.environ["TOKENIZERS_PARALLELISM"] = "false"
		data_augmentation = config_file["global"]["data_augmentation"]
		self.freeze_embeddings = config_file["global"]["freeze_embeddings"]
		self.freeze_lang_embeddings = config_file["global"]["freeze_lang_embeddings"]
		self.balance_class_weights = config_file["global"]["balance_class_weights"]
		include_lang_metadata = config_file["global"]["include_lang_metadata"]
		lang_emb_dim = config_file["global"]["lang_emb_dim"]
		linear_layers = config_file["global"]["linear_layers"]
		linear_layers_hidden_size = config_file["global"]["linear_layers_hidden_size"]
		emb_dim = config_file["global"]["emb_dim"]
		if architecture == "lstm":
			add_attention_layer = config_file["architectures"][architecture]["add_attention_layer"]
			lstm_hidden_size = config_file["architectures"][architecture]["lstm_hidden_size"]
			num_lstm_layers = config_file["architectures"][architecture]["num_lstm_layers"]
			lstm_dropout = config_file["architectures"][architecture]["lstm_dropout"]
			bidirectional = config_file["architectures"][architecture]["bidirectional"]
		elif architecture == "gru":
			add_attention_layer = config_file["architectures"][architecture]["add_attention_layer"]
			hidden_size = config_file["architectures"][architecture]["hidden_size"]
			num_layers = config_file["architectures"][architecture]["num_layers"]
			dropout = config_file["architectures"][architecture]["dropout"]
			bidirectional = config_file["architectures"][architecture]["bidirectional"]
		elif architecture == "transformer":
			num_heads = config_file["architectures"][architecture]["num_heads"]
			num_transformers_layers = config_file["architectures"][architecture]["num_transformers_layers"]
		elif architecture == "cnn":
			dropout = config_file["architectures"][architecture]["dropout"]
			cnn_scale = config_file["architectures"][architecture]["cnn_scale"]
			add_attention_layer = config_file["architectures"][architecture]["add_attention_layer"]
			hidden_size = config_file["architectures"][architecture]["hidden_size"]
			kernel_size = config_file['architectures'][architecture]["kernel_size"]
			positional_embeddings = config_file['architectures'][architecture]["positional_embeddings"]
			num_heads = config_file["architectures"][architecture]["num_heads"]
			num_cnn_layers = config_file["architectures"][architecture]["num_cnn_layers"]



		# First we prepare the corpus
		now = datetime.datetime.now()
		self.device = device
		if self.device != "cpu":
			device_name = torch.cuda.get_device_name(self.device)
			print(f"Device name: {device_name}")
		self.workers = workers
		self.timestamp = now.strftime("%d-%m-%Y_%H:%M:%S")
		self.all_dataset_on_device = False
		print("Loading data")
		if use_pretrained_embeddings:
			create_vocab = False
			self.tokenizer = AutoTokenizer.from_pretrained(base_model_name)
		else:
			if use_bert_tokenizer:
				create_vocab = False
				self.tokenizer = AutoTokenizer.from_pretrained(base_model_name)
			else:
				create_vocab = True
				self.tokenizer = None

		self.train_path = train_path
		self.test_path = test_path
		self.dev_path = dev_path
		self.fine_tune = fine_tune
		self.output_dir = output_dir
		self.use_pretrained_embeddings = use_pretrained_embeddings
		self.base_model_name = base_model_name


		self.data_augmentation = data_augmentation
		self.train_dataloader = datafy.CustomTextDataset("train",
													train_path=train_path,
													test_path=test_path,
													dev_path=dev_path,
													device=self.device,
													delimiter="£",
													output_dir=output_dir,
													create_vocab=create_vocab,
													use_pretrained_embeddings=use_pretrained_embeddings,
													debug=self.debug,
													data_augmentation=self.data_augmentation,
													tokenizer_name=base_model_name,
													use_bert_tokenizer=use_bert_tokenizer)
		self.test_dataloader = datafy.CustomTextDataset(mode="test",
												   train_path=train_path,
												   test_path=test_path,
													dev_path=dev_path,
												   device=self.device,
												   delimiter="£",
												   output_dir=output_dir,
												   create_vocab=False,
												   input_vocab=self.train_dataloader.datafy.input_vocabulary,
												   lang_vocab=self.train_dataloader.datafy.lang_vocabulary,
													use_pretrained_embeddings=use_pretrained_embeddings,
													debug=self.debug,
													data_augmentation=self.data_augmentation,
													tokenizer_name=base_model_name,
													use_bert_tokenizer=use_bert_tokenizer)

		self.dev_dataloader = datafy.CustomTextDataset(mode="dev",
												   train_path=train_path,
												   test_path=test_path,
													dev_path=dev_path,
												   device=self.device,
												   delimiter="£",
												   output_dir=output_dir,
												   create_vocab=False,
												   input_vocab=self.train_dataloader.datafy.input_vocabulary,
												   lang_vocab=self.train_dataloader.datafy.lang_vocabulary,
													use_pretrained_embeddings=use_pretrained_embeddings,
													debug=self.debug,
													data_augmentation=self.data_augmentation,
													tokenizer_name=base_model_name,
													use_bert_tokenizer=use_bert_tokenizer)

		self.loaded_test_data = DataLoader(self.test_dataloader,
										   batch_size=batch_size,
										   shuffle=False,
										   num_workers=self.workers,
										   pin_memory=False,
										   drop_last=True)
		self.loaded_train_data = DataLoader(self.train_dataloader,
											batch_size=batch_size,
											shuffle=True,
											num_workers=self.workers,
											pin_memory=False,
										   drop_last=True)
		self.loaded_dev_data = DataLoader(self.dev_dataloader,
											batch_size=batch_size,
											shuffle=True,
											num_workers=self.workers,
											pin_memory=False,
										   drop_last=True)



		self.output_dir = output_dir
		# On crée l'output dir:
		os.makedirs(f"{self.output_dir}/models/.tmp", exist_ok=True)
		os.makedirs(f"{self.output_dir}/best", exist_ok=True)

		print(f"Number of train examples: {len(self.train_dataloader.datafy.train_padded_examples)}")
		print(f"Number of test examples: {len(self.test_dataloader.datafy.test_padded_examples)}")
		print(f"Total length of examples (with padding): {self.train_dataloader.datafy.max_length_examples}")
		self.input_vocab = self.train_dataloader.datafy.input_vocabulary
		self.reverse_input_vocab = {v: k for k, v in self.input_vocab.items()}

		self.lang_vocab = self.train_dataloader.datafy.lang_vocabulary





		self.target_classes = self.train_dataloader.datafy.target_classes
		self.reverse_target_classes = self.train_dataloader.datafy.reverse_target_classes

		self.corpus_size = self.train_dataloader.__len__()
		self.steps = self.corpus_size // batch_size

		self.test_steps = self.test_dataloader.__len__() // batch_size
		self.tgt_PAD_IDX = self.target_classes["[PAD]"]
		self.epochs = epochs
		self.batch_size = batch_size
		self.output_dim = len(self.target_classes)
		self.include_lang_metadata = include_lang_metadata
		self.best_model = ""
		self.input_dim = len(self.input_vocab)


		# Ici on choisit quelle architecture on veut tester. À faire: CNN et RNN

		weights = torch.load("aquilign/segmenter/embeddings.npy")

		if architecture == "transformer":
			self.model = models.TransformerModel(input_dim=self.input_dim,
												 emb_dim=emb_dim,
												 num_heads=num_heads,
												 num_layers=num_transformers_layers,
												 output_dim=self.output_dim,
												 num_langs=len(self.lang_vocab),
												 lang_emb_dim=lang_emb_dim,
												 include_lang_metadata=True,
												 device=device,
												 linear_layers=linear_layers,
												 linear_layers_hidden_size=linear_layers_hidden_size,
												 load_pretrained_embeddings=use_pretrained_embeddings,
												 use_bert_tokenizer=use_bert_tokenizer,
												 pretrained_weights=weights)
		elif architecture == "lstm":
			self.model = models.LSTM_Encoder(input_dim=self.input_dim,
											 emb_dim=emb_dim,
											 bidirectional=bidirectional,
											 lstm_dropout=lstm_dropout,
											 positional_embeddings=False,
											 device=self.device,
											 lstm_hidden_size=lstm_hidden_size,
											 batch_size=batch_size,
											 num_langs=len(self.lang_vocab),
											 num_lstm_layers=num_lstm_layers,
											 include_lang_metadata=include_lang_metadata,
											 out_classes=self.output_dim,
											 attention=add_attention_layer,
											 lang_emb_dim=lang_emb_dim,
											 load_pretrained_embeddings=use_pretrained_embeddings,
											 pretrained_weights=weights,
											 linear_layers=linear_layers,
											 linear_layers_hidden_size=linear_layers_hidden_size,
											 use_bert_tokenizer=use_bert_tokenizer)
		elif architecture == "gru":
			self.model = models.GRU_Encoder(input_dim=self.input_dim,
											 emb_dim=emb_dim,
											 bidirectional=bidirectional,
											 dropout=dropout,
											 positional_embeddings=False,
											 device=self.device,
											 hidden_size=hidden_size,
											 batch_size=batch_size,
											 num_langs=len(self.lang_vocab),
											 num_layers=num_layers,
											 include_lang_metadata=include_lang_metadata,
											 out_classes=self.output_dim,
											 attention=add_attention_layer,
											 lang_emb_dim=lang_emb_dim,
											 load_pretrained_embeddings=use_pretrained_embeddings,
											 pretrained_weights=weights
					)
		elif architecture == "cnn":
			self.model = models.CnnEncoder(input_dim=self.input_dim,
									   emb_dim=emb_dim,
									   dropout=dropout,
									   kernel_size=kernel_size,
									   positional_embeddings=positional_embeddings,
									   device=self.device,
									   hidden_size=hidden_size,
									   num_langs=len(self.lang_vocab),
									   num_conv_layers=num_cnn_layers,
									   include_lang_metadata=include_lang_metadata,
									   out_classes=self.output_dim,
									   attention=add_attention_layer,
									   lang_emb_dim=lang_emb_dim,
									   load_pretrained_embeddings=use_pretrained_embeddings,
										use_bert_tokenizer=use_bert_tokenizer,
										linear_layers_hidden_size=linear_layers_hidden_size,
										linear_layers=linear_layers,
									   pretrained_weights=weights,
										cnn_scale=cnn_scale
									   )
		self.architecture = architecture
		self.model.to(self.device)
		self.optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, self.model.parameters()), lr=lr)
		self.scheduler = torch.optim.lr_scheduler.ExponentialLR(self.optimizer, gamma=0.9)
		self.use_pretrained_embeddings = use_pretrained_embeddings

		# Les classes étant distribuées de façons déséquilibrée, on donne + d'importance à la classe <SB>
		# qu'aux deux autres pour le calcul de la loss. On désactive pour l'instant
		if self.balance_class_weights:
			weights = self.train_dataloader.datafy.target_weights.to(self.device)
			self.criterion = torch.nn.CrossEntropyLoss(weight=weights, ignore_index=self.tgt_PAD_IDX)
		else:
			self.criterion = torch.nn.CrossEntropyLoss(ignore_index=self.tgt_PAD_IDX)
		print(self.model.__repr__())
		self.accuracies = []
		self.results = []

	def save_model(self, epoch):
		torch.save(self.model.state_dict(), f"{self.output_dir}/models/.tmp/model_segmenter_{self.architecture}_{epoch}.pt")
		print(f"Model saved to {self.output_dir}/models/.tmp/model_segmenter_{self.architecture}_{epoch}.pt")


	def get_best_model(self):
		"""
		We choose the best model based on a weighted average of precision and recall.
		"""
		weighted_averages = []
		for epoch, result in enumerate(self.results):
			recall = result["recall"][1]
			precision = result["precision"][1]
			weighted = (precision + (recall*2) ) / 3
			weighted_averages.append(weighted.item())

		max_average = max(weighted_averages)
		best_epoch = weighted_averages.index(max_average)
		print(f"Best model: {best_epoch} with {max_average} weighted precision and recall.")
		models = glob.glob(f"{self.output_dir}/models/.tmp/model_segmenter_{self.architecture}_*.pt")
		try:
			os.mkdir(f"{self.output_dir}/models/best/{self.architecture}")
		except OSError:
			pass
		for model in models:
			if model == f"{self.output_dir}/models/.tmp/model_segmenter_{self.architecture}_{best_epoch}.pt":
				shutil.copy(model, f"{self.output_dir}/models/best/{self.architecture}/best.pt")
				print(f"Saving best model to {self.output_dir}/models/best/{self.architecture}/best.pt")
			else:
				continue
				os.remove(model)
		self.best_model = f"{self.output_dir}/models/best/{self.architecture}/best.pt"

	def train(self, clip=0.1):
		# Ici on va faire en sorte que les plongements de mots ne soient pas entraînables, si c'est des plongements pré-entraînés
		# Une possibilité serait de dégeler les paramètres en fin d'entraînement, quelques epochs avant la fin (3-4?)
		if self.use_pretrained_embeddings:
			if self.freeze_embeddings:
				for param in self.model.embedding.parameters():
					param.requires_grad = False

		# Idem pour les plongements de langue.
		if self.include_lang_metadata:
			if self.freeze_lang_embeddings:
				for param in self.model.lang_embedding.parameters():
					param.requires_grad = False
		utils.remove_file(f"{self.output_dir}/accuracies.txt")
		utils.remove_files(f"{self.output_dir}/models/.tmp/model_segmenter_{self.architecture}_*.pt")
		print("Starting training")
		os.makedirs(f"{self.output_dir}/models/best", exist_ok=True)
		torch.save(self.input_vocab, f"{self.output_dir}/models/best/vocab.voc")
		print("Evaluating randomly initiated model")
		self.evaluate()
		torch.save(self.model, f"{self.output_dir}/models/model_orig.pt")
		for epoch in range(self.epochs):
			self.model.train()
			epoch_number = epoch + 1
			last_epoch = epoch == range(self.epochs)[-1]
			print(f"Epoch {str(epoch_number)}")
			for examples, langs, targets in tqdm.tqdm(self.loaded_train_data, unit_scale=self.batch_size):
				# Shape [batch_size, max_length]
				# tensor_examples = examples.to(self.device)
				# Shape [batch_size, max_length]
				# tensor_targets = targets.to(self.device)
				if not self.all_dataset_on_device:
					examples = examples.to(self.device)
					targets = targets.to(self.device)
					langs = langs.to(self.device)

				# https://pytorch.org/tutorials/recipes/recipes/tuning_guide.html
				# for param in self.model.parameters():
					# param.grad = None
				self.optimizer.zero_grad()
				# Shape [batch_size, max_length, output_dim]
				output = self.model(examples, langs)
				# output_dim = output.shape[-1]
				# Shape [batch_size*max_length, output_dim]
				output = output.view(-1, self.output_dim)
				# Shape [batch_size*max_length]
				tgt = targets.view(-1)

				# output = [batch size * tgt len - 1, output dim]
				# tgt = [batch size * tgt len - 1]
				loss = self.criterion(output, tgt)
				loss.backward()
				torch.nn.utils.clip_grad_norm_(self.model.parameters(), clip)
				self.optimizer.step()

			# self.model.eval()
			self.scheduler.step()
			self.evaluate(last_epoch=last_epoch)
			self.save_model(epoch_number)
		self.get_best_model()
		self.evaluate_best_model()
		self.evaluate_best_model_per_lang()

	def evaluate_best_model(self):
		"""
				Cette fonction produit les métriques d'évaluation (justesse, précision, rappel)
				"""
		print("Evaluating best model on test data")
		all_preds = []
		all_targets = []
		all_examples = []
		self.model.load_state_dict(torch.load(self.best_model, weights_only=True))
		self.model.eval()
		for examples, langs, targets in tqdm.tqdm(self.loaded_test_data, unit_scale=self.batch_size):
			if not self.all_dataset_on_device:
				tensor_examples = examples.to(self.device)
				tensor_langs = langs.to(self.device)
				tensor_target = targets.to(self.device)
			with torch.no_grad():
				# On prédit. La langue est toujours envoyée même si elle n'est pas traitée par le modèle, pour des raisons de simplicité
				preds = self.model(tensor_examples, tensor_langs)
				all_preds.append(preds)
				all_targets.append(targets)
				all_examples.append(examples)

		# On crée une seul vecteur, en concaténant tous les exemples sur la dimension 0 (= chaque exemple individuel)
		cat_preds = torch.cat(all_preds, dim=0)
		cat_targets = torch.cat(all_targets, dim=0)
		cat_examples = torch.cat(all_examples, dim=0)
		ambiguity = eval.compute_ambiguity_metrics(tokens=cat_examples,
												   labels=cat_targets,
												   predictions=cat_preds,
												   id_to_word=self.reverse_input_vocab,
												   word_to_id=self.input_vocab,
												   output_dir = self.output_dir)
		results = eval.compute_metrics(predictions=cat_preds,
									   labels=cat_targets,
									   examples=cat_examples,
									   id_to_word=self.reverse_input_vocab,
									   idx_to_class=self.reverse_target_classes,
									   padding_idx=self.tgt_PAD_IDX,
									   batch_size=self.batch_size,
									   last_epoch=True,
									   tokenizer=self.tokenizer)



		recall = ["Recall", results["recall"][0], results["recall"][1]]
		precision = ["Precision", results["precision"][0], results["precision"][1]]
		f1 = ["F1", results["f1"][0], results["f1"][1]]
		header = ["", "Segment Content", "Segment Boundary"]
		print(f"Results for all langs:")
		utils.format_results(results=[precision, recall, f1], header=header)

	def evaluate_best_model_per_lang(self):
		"""
				Cette fonction produit les métriques d'évaluation (justesse, précision, rappel)
				"""
		print("Evaluating best model on test data")

		# On crée un dernier dataloader: un dictionnaire avec division des langues pour avoir des résultats par langue.
		loaded_test_data_per_lang = {}
		batch_size = self.batch_size
		for lang in self.lang_vocab:
			current_dataloader = datafy.CustomTextDataset(mode="test",
														  train_path=self.train_path,
														  test_path=self.test_path,
														  dev_path=self.dev_path,
														  device=self.device,
														  delimiter="£",
														  output_dir=self.output_dir,
														  create_vocab=False,
														  input_vocab=self.train_dataloader.datafy.input_vocabulary,
														  lang_vocab=self.train_dataloader.datafy.lang_vocabulary,
														  use_pretrained_embeddings=self.use_pretrained_embeddings,
														  debug=self.debug,
														  data_augmentation=self.data_augmentation,
														  filter_by_lang=lang,
														 tokenizer_name=self.base_model_name)
			loaded_test_data_per_lang[lang] = DataLoader(current_dataloader,
															  batch_size=batch_size,
															  shuffle=False,
															  num_workers=self.workers,
															  pin_memory=False,
															  drop_last=True)



		self.model.load_state_dict(torch.load(self.best_model, weights_only=True))
		self.model.eval()
		results_per_lang = {}
		for lang in self.lang_vocab:
			print(f"Testing {lang}")
			all_preds = []
			all_targets = []
			all_examples = []
			for examples, langs, targets in tqdm.tqdm(loaded_test_data_per_lang[lang], unit_scale=batch_size):
				if not self.all_dataset_on_device:
					tensor_examples = examples.to(self.device)
					tensor_langs = langs.to(self.device)
					tensor_target = targets.to(self.device)
				with torch.no_grad():
					# On prédit. La langue est toujours envoyée même si elle n'est pas traitée par le modèle, pour des raisons de simplicité
					preds = self.model(tensor_examples, tensor_langs)
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
										   id_to_word=self.reverse_input_vocab,
										   idx_to_class=self.reverse_target_classes,
										   padding_idx=self.tgt_PAD_IDX,
										   batch_size=batch_size,
										   last_epoch=False,
										   tokenizer=self.tokenizer)
			results_per_lang[lang] = results

			recall = ["Recall", results["recall"][0], results["recall"][1]]
			precision = ["Precision", results["precision"][0], results["precision"][1]]
			f1 = ["F1", results["f1"][0], results["f1"][1]]
			header = ["", "Segment Content", "Segment Boundary"]
			print(f"Results for {lang}:")
			utils.format_results(results=[precision, recall, f1], header=header)




	def evaluate(self, loss_calculation:bool=False, last_epoch:bool=False):
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
		self.model.eval()
		for examples, langs, targets in tqdm.tqdm(self.loaded_dev_data, unit_scale=self.batch_size):
			# https://discuss.pytorch.org/t/should-we-set-non-blocking-to-true/38234/3
			# Timer.start_timer("preds")
			if not self.all_dataset_on_device:
				tensor_examples = examples.to(self.device)
				tensor_langs = langs.to(self.device)
				tensor_target = targets.to(self.device)
			with torch.no_grad():
				# On prédit. La langue est toujours envoyée même si elle n'est pas traitée par le modèle, pour des raisons de simplicité
				preds = self.model(tensor_examples, tensor_langs)
				all_preds.append(preds)
				all_targets.append(targets)
				all_examples.append(examples)

				if loss_calculation:
					output_dim = preds.shape[-1]
					output = preds.contiguous().view(-1, output_dim)
					tgt = tensor_target.contiguous().view(-1)
					loss = self.criterion(output, tgt)

		# On supprime les batchs:
		cat_preds = torch.cat(all_preds, dim=0) # [num_examples, max_dim, num_classes]
		cat_targets = torch.cat(all_targets, dim=0) # [num_examples, max_dim]
		cat_examples = torch.cat(all_examples, dim=0) # [num_examples, max_dim]
		results = eval.compute_metrics(predictions=cat_preds,
									   labels=cat_targets,
									   examples=cat_examples,
									   id_to_word=self.reverse_input_vocab,
									   idx_to_class=self.reverse_target_classes,
									   padding_idx=self.tgt_PAD_IDX,
									   batch_size=self.batch_size,
									   last_epoch=last_epoch,
									   tokenizer=self.tokenizer)
		self.results.append(results)

		recall = ["Recall", results["recall"][0], results["recall"][1]]
		precision = ["Precision", results["precision"][0], results["precision"][1]]
		f1 = ["F1", results["f1"][0], results["f1"][1]]
		header = ["", "Segment Content", "Segment Boundary"]
		print(f"Results for all langs:")
		utils.format_results(results=[precision, recall, f1], header=header)
		return (recall, precision, f1)
