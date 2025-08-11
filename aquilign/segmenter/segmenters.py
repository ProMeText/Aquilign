import re
import torch
import aquilign.segmenter.trainer as trainer




if __name__ == '__main__':
	device = sys.argv[4]
	train_path = sys.argv[1]
	test_path = sys.argv[2]
	output_dir = sys.argv[3]
	# test_path = "/home/mgl/Bureau/Travail/projets/alignement/alignement_multilingue/multilingual-segmentation-dataset/data/Multilingual_Aegidius/segmented/split/multilingual/test.json"
	# train_path = "/home/mgl/Bureau/Travail/projets/alignement/alignement_multilingue/multilingual-segmentation-dataset/data/Multilingual_Aegidius/segmented/split/multilingual/train.json"
	# output_dir = "/home/mgl/Documents/lstm/"
	trainer = trainer.Trainer(architecture="lstm",
							  epochs=1,
							  batch_size=32,
							  lr=0.0005,
							  fine_tune=False,
							  device=device,
							  train_path=train_path,
							  test_path=test_path,
							  output_dir=output_dir,
							  workers=8
							  )

	trainer.train()
