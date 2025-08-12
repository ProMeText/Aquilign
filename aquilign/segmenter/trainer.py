import re
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

class Trainer:
	def  __init__(self,
				  config_file,
				  architecture,
				  epochs,
				  lr,
				  device,
				  batch_size,
				  train_path,
				  test_path,
				  fine_tune:bool,
				  output_dir:str,
				  workers:int,
				  include_lang_metadata:bool,
				  add_attention_layer:bool,
				  lstm_dropout:float):
		# First we prepare the corpus
		now = datetime.datetime.now()
		self.device = device
		if self.device != "cpu":
			device_name = torch.cuda.get_device_name(self.device)
			print(f"Device name: {device_name}")
		self.workers = workers
		max_length = 300
		self.timestamp = now.strftime("%d-%m-%Y_%H:%M:%S")
		if fine_tune:
			input_vocab = torch.load(pretrained_params.get("vocab"))
		else:
			input_vocab = None
		self.all_dataset_on_device = False
		print("Loading data")
		train_dataloader = datafy.CustomTextDataset("train", train_path, test_path, fine_tune, input_vocab, max_length, self.device, self.all_dataset_on_device, "£")
		test_dataloader = datafy.CustomTextDataset("test", train_path, test_path, fine_tune, input_vocab, max_length, self.device, self.all_dataset_on_device, "£")

		self.loaded_test_data = DataLoader(test_dataloader,
										   batch_size=batch_size,
										   shuffle=False,
										   num_workers=8,
										   pin_memory=False,
										   drop_last=True)
		self.loaded_train_data = DataLoader(train_dataloader,
											batch_size=batch_size,
											shuffle=True,
											num_workers=self.workers,
											pin_memory=False,
										   drop_last=True)

		self.output_dir = output_dir
		# On crée l'output dir:
		os.makedirs(f"{self.output_dir}/models/.tmp", exist_ok=True)
		os.makedirs(f"{self.output_dir}/best", exist_ok=True)

		print(f"Number of train examples: {len(train_dataloader.datafy.train_padded_examples)}")
		print(f"Number of test examples: {len(test_dataloader.datafy.test_padded_examples)}")
		print(f"Total length of examples (with padding): {train_dataloader.datafy.max_length_examples}")

		self.input_vocab = train_dataloader.datafy.input_vocabulary
		self.reverse_input_vocab = {v: k for k, v in self.input_vocab.items()}
		self.lang_vocab = train_dataloader.datafy.lang_vocabulary
		self.target_classes = train_dataloader.datafy.target_classes
		self.reverse_target_vocab = {v: k for k, v in self.target_classes.items()}

		self.corpus_size = train_dataloader.__len__()
		self.steps = self.corpus_size // batch_size

		self.test_steps = test_dataloader.__len__() // batch_size
		self.tgt_PAD_IDX = self.target_classes["<PAD>"]
		self.epochs = epochs
		self.batch_size = batch_size
		self.output_dim = len(self.target_classes)


		if False:
			self.pretrained_model = pretrained_params.get('model', None)
			self.pretrained_vocab = pretrained_params.get('vocab', None)
			self.input_vocab = train_dataloader.datafy.input_vocabulary
			self.input_dim = len(self.input_vocab)
			torch.save(self.input_vocab, f"{output_dir}/vocab.voc")
			if self.device == 'cpu':
				self.pre_trained_model = torch.load(self.pretrained_model, map_location=self.device)
			else:
				self.pre_trained_model = torch.load(self.pretrained_model).to(self.device)
			self.pretrained_vocab = torch.load(self.pretrained_vocab)
			pre_trained_weights = self.pre_trained_model.encoder.tok_embedding.weight
			embs_dim = pre_trained_weights.shape[1]
			self.reverse_input_vocab = {v: k for k, v in self.input_vocab.items()}

			# We create the updated embs:
			# First we create randomly initiated tensors corresponding to the number of new chars in the new dataset
			number_new_chars = len(self.input_vocab) - len(self.pretrained_vocab)
			new_vectors = torch.zeros(number_new_chars, embs_dim).to(self.device)

			# We then add the new vectors to the pre-trained weights
			updated_vectors = torch.cat((pre_trained_weights, new_vectors), 0)
			# We then take the pre-trained model and modify its embedding layer to match
			# new + old vocabulary
			self.model = self.pre_trained_model
			self.model.encoder.tok_embedding = nn.Embedding.from_pretrained(updated_vectors)

		else:
			self.input_dim = len(self.input_vocab)
			if architecture == "cnn":
				EMB_DIM = 256
				HID_DIM = 256  # each conv. layer has 2 * hid_dim filters
				ENC_LAYERS = 10  # number of conv. blocks in encoder
				ENC_KERNEL_SIZE = kernel_size  # must be odd!
				ENC_DROPOUT = 0.25
				self.enc = models.CnnEncoder(self.input_dim, EMB_DIM, HID_DIM, ENC_LAYERS, ENC_KERNEL_SIZE, ENC_DROPOUT,
										  self.device)
				self.dec = models.LinearDecoder(EMB_DIM, self.output_dim)
				self.model = seq2seq.Seq2Seq(self.enc, self.dec)
			elif architecture == "rnn":
				pass
			elif architecture == "lstm":
				self.model = models.LSTM_Encoder(input_dim=self.input_dim,
												 emb_dim=300,
												 bidirectional_lstm=True,
												 lstm_dropout=lstm_dropout,
												 positional_embeddings=False,
												 device=self.device,
												 lstm_hidden_size=32,
												 batch_size=batch_size,
												 num_langs=len(self.lang_vocab),
												 num_lstm_layers=1,
												 include_lang_metadata=include_lang_metadata,
												 out_classes=self.output_dim,
												 attention=add_attention_layer,
												 lang_emb_dim=32
						)
		self.architecture = architecture
		self.model.to(self.device)
		self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
		self.scheduler = torch.optim.lr_scheduler.ExponentialLR(self.optimizer, gamma=0.9)
		self.criterion = torch.nn.CrossEntropyLoss(ignore_index=self.tgt_PAD_IDX)
		print(self.model.__repr__())
		self.accuracies = []

	def save_model(self, epoch):
		torch.save(self.model, f"{self.output_dir}/models/.tmp/model_segmenter_{self.architecture}_{epoch}.pt")

	def get_best_model(self):
		print(self.accuracies)
		best_epoch_accuracy = self.accuracies.index(max(self.accuracies))
		print(f"Best model: {best_epoch_accuracy}.")
		models = glob.glob(f"{self.output_dir}/models/.tmp/model_segmenter_{self.architecture}_*.pt")
		for model in models:
			if model == f"{self.output_dir}/models/.tmp/model_segmenter_{self.architecture}_{best_epoch_accuracy}.pt":
				shutil.copy(model, f"{self.output_dir}/best.pt")
			else:
				os.remove(model)
		print(f"Saving best model to {self.output_dir}/best.pt")

	def train(self, clip=0.1):
		utils.remove_file(f"{self.output_dir}/accuracies.txt")
		print("Starting training")
		torch.save(self.input_vocab, f"{self.output_dir}/vocab.voc")
		print("Evaluating randomly intiated model")
		self.evaluate()
		torch.save(self.model, f"{self.output_dir}/model_orig.pt")
		self.model.train()
		for epoch in range(self.epochs):
			epoch_number = epoch + 1
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
			self.evaluate()
			self.save_model(epoch_number)
		self.get_best_model()

	def evaluate(self, loss_calculation=False):
		"""
		Réécrire la fonction pour comparer directement target et prédiction pour
		produire l'accuracy.
		"""
		print("Evaluating model on test data")
		debug = False
		epoch_accuracy = []
		epoch_loss = []
		# Timer = utils.Timer()
		all_preds = []
		all_targets = []
		for examples, langs, targets in tqdm.tqdm(self.loaded_test_data, unit_scale=self.batch_size):
			# https://discuss.pytorch.org/t/should-we-set-non-blocking-to-true/38234/3
			# Timer.start_timer("preds")
			if not self.all_dataset_on_device:
				tensor_examples = examples.to(self.device)
				tensor_langs = langs.to(self.device)
				tensor_target = targets.to(self.device)
			with torch.no_grad():
				preds = self.model(tensor_examples, tensor_langs)
				all_preds.append(preds)
				all_targets.append(targets)

				if loss_calculation:
					output_dim = preds.shape[-1]
					output = preds.contiguous().view(-1, output_dim)
					tgt = tensor_target.contiguous().view(-1)
					loss = self.criterion(output, tgt)

		# On crée une seul vecteur, en concaténant tous les exemples sur la dimension 0 (= chaque exemple individuel)
		cat_preds = torch.cat(all_preds, dim=0)
		cat_targets = torch.cat(all_targets, dim=0)
		results = eval.compute_metrics(cat_preds, cat_targets, self.tgt_PAD_IDX)
		self.accuracies.append(results["accuracy"]["accuracy"])
