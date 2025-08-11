import re
import torch
import sys
import json
with open(sys.argv[1], "r") as input_json:
	config_file = json.load(input_json)

# Gestion des imports
if config_file["import"] != "":
	sys.path.append(config_file["import"])
import aquilign.segmenter.trainer as trainer




if __name__ == '__main__':
	# test_path = "/home/mgl/Bureau/Travail/projets/alignement/alignement_multilingue/multilingual-segmentation-dataset/data/Multilingual_Aegidius/segmented/split/multilingual/test.json"
	# train_path = "/home/mgl/Bureau/Travail/projets/alignement/alignement_multilingue/multilingual-segmentation-dataset/data/Multilingual_Aegidius/segmented/split/multilingual/train.json"
	# output_dir = "/home/mgl/Documents/lstm/"
	architecture = sys.argv[1]
	epochs = config_file[architecture]["epochs"]
	batch_size = config_file[architecture]["batch_size"]
	lr = config_file["architectures"][architecture]["lr"]
	device = config_file["global"]["device"]
	workers = config_file["global"]["workers"]
	train_path = config_file["global"]["train"]
	test_path = config_file["global"]["test"]
	output_dir = config_file["global"]["out_dir"]
	trainer = trainer.Trainer(config_file=config_file,
							  architecture=architecture,
							  epochs=epochs,
							  batch_size=batch_size,
							  lr=lr,
							  fine_tune=False,
							  device=device,
							  train_path=train_path,
							  test_path=test_path,
							  output_dir=output_dir,
							  workers=workers
							  )

	trainer.train()
