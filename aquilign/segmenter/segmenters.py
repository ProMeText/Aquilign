import re
import torch
import sys
import json
with open(sys.argv[1], "r") as input_json:
	config_file = json.load(input_json)

# Gestion des imports
if config_file["global"]["import"] != "":
	sys.path.append(config_file["global"]["import"])
import aquilign.segmenter.trainer as trainer




if __name__ == '__main__':
	# test_path = "/home/mgl/Bureau/Travail/projets/alignement/alignement_multilingue/multilingual-segmentation-dataset/data/Multilingual_Aegidius/segmented/split/multilingual/test.json"
	# train_path = "/home/mgl/Bureau/Travail/projets/alignement/alignement_multilingue/multilingual-segmentation-dataset/data/Multilingual_Aegidius/segmented/split/multilingual/train.json"
	# output_dir = "/home/mgl/Documents/lstm/"
	trainer = trainer.Trainer(config_file=config_file)

	trainer.train()
