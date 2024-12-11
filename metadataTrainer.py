# -*- coding: utf-8 -*-
import sys
from transformers import BertTokenizer, Trainer, TrainingArguments, AutoModelForTokenClassification, set_seed, TrainerCallback, BertForTokenClassification, BertConfig, PretrainedConfig
from accelerate import Accelerator, DataLoaderConfiguration
import aquilign.preproc.tok_trainer_functions as trainer_functions
import aquilign.preproc.eval as evaluation
import aquilign.preproc.utils as utils
import random
import re
import os
import json
import glob
import shutil
import argparse
import aquilign.preproc.metadataModel as metadataModel


## script for the training of the text tokenizer : identification of tokens (label 1) which will be used to split the text
## produces folder with models (best for each epoch) and logs


## usage : python tok_trainer.py model_name train_file.txt dev_file.txt num_train_epochs batch_size logging_steps
## where :
# model_name is the full name of the model (same name for model and tokenizer)
# train_file.txt is the file with the sentences and words of interest are identified  (words are identified with $ after the line)
# which will be used for training
## ex. : uoulentiers mais il nen est pas encor temps. Certes fait elle si$mais£Certes
# dev_file.txt is the file with the sentences and words of interest which will be used for eval
# num_train_epochs : the number of epochs we want to train (ex : 10)
# batch_size : the batch size (ex : 8)
# logging_steps : the number of logging steps (ex : 50)

class SaveModelAndConfigCallback(TrainerCallback):
    def __init__(self, save_dir: str):
        self.save_dir = save_dir

    def on_epoch_end(self, args, state, control, **kwargs):
        # Check if we need to save the model at the end of each epoch
        output_dir = os.path.join(self.save_dir, f"checkpoint-{state.global_step}")
        os.makedirs(output_dir, exist_ok=True)
        kwargs['model'].bert.save_pretrained(output_dir)  # Save the model weights
        kwargs['model'].bert.config.save_pretrained(output_dir)  # Save the config.json
        print(f"Model and config saved at {output_dir}")
        return control

class CustomTrainer(Trainer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def save_model_and_config(self, output_dir):
        # Save the model and config file to the given output directory
        self.model.bert.save_pretrained(output_dir)
        self.model.bert.config.save_pretrained(output_dir)
        print(self.model.metadata_embedding)
        print(f"Saving config to {output_dir}")
        exit(0)

    def on_epoch_begin(self, args, state, control, **kwargs):
        # You can customize what happens at the end of an epoch here
        # Save the model and config after each epoch
        output_dir = os.path.join(args.output_dir, f"checkpoint-{state.global_step}")
        os.makedirs(output_dir, exist_ok=True)
        self.save_model_and_config(output_dir)
        return control


# function which produces the train, which first gets texts, transforms them into tokens and labels, then trains model with the specific given arguments
def training_trainer(modelName, datasets, num_train_epochs, batch_size, logging_steps,
                     keep_punct=True, freeze_metadata=False, device="cpu", train_name="train"):
    
    
    new_best_path = f"tokenisation_training_results/results_{train_name}/best"
    if os.path.isdir(new_best_path):
        print(f"This script won't perform dangerous recursive dir deletions. "
              f"Please remove tokenisation_training_results/results_{train_name}/best (target dir of training) and relaunch script. "
              f"Exiting")
        exit(0)
    
    config = BertConfig.from_pretrained("google-bert/bert-base-multilingual-cased")
    config.num_labels = 3  # Exemple : 3 classes
    config.num_metadata_features = 5
    config.freeze_metadata = freeze_metadata
    config.name_or_path = "google-bert/bert-base-multilingual-cased"
    model = metadataModel.BertWithMetadata(config)
    tokenizer = BertTokenizer.from_pretrained(modelName, max_length=10)
    


    datasets = {"train": "data/tests/it/it.txt",
                 "eval": "data/tests/it/it.txt",
                 "dev": "data/tests/it/it.txt"}
    
    datasets = {"train": "data/tokenisation/*/*train.txt",
                "eval": "data/tokenisation/*/*eval.txt",
                "dev": "data/tokenisation/*/*dev.txt"}
    
    
    
    
    training_files = glob.glob(datasets['train'])
    dev_files = glob.glob(datasets['dev'])
    eval_files = glob.glob(datasets['eval'])
    print(training_files)
    train_lines = []
    dev_lines = []
    for file in training_files:
        print(file)
        with open(file, "r") as train_file:
            lang = file.split("/")[-2]
            train_lines.extend([(item.replace("\n", ""), lang) for item in train_file.readlines()])
            if keep_punct is False:
                train_lines = [(utils.remove_punctuation(line), lang) for line, lang in train_lines]

    for file in dev_files:
        with open(file, "r") as dev_file:
            lang = file.split("/")[-2]
            dev_lines.extend([(item.replace("\n", ""), lang) for item in dev_file.readlines()])
            if keep_punct is False:
                dev_lines = [(utils.remove_punctuation(line), lang) for line, lang in dev_lines]
    
    random.shuffle(train_lines)
    random.shuffle(dev_lines)
    

    eval_lines = {}
    for file in eval_files:
        with open(file, "r") as eval_files:
            lang = file.split("/")[-2]
            as_lines = [(item.replace("\n", ""), lang) for item in eval_files.readlines()]
            try:
                eval_lines[lang].extend(as_lines)
            except KeyError:
                eval_lines[lang] = as_lines
            if keep_punct is False:
                eval_lines = [(utils.remove_punctuation(line), lang) for line, lang in eval_lines]
        random.shuffle(eval_lines[lang])
        
    # We create full eval corpus too
    full_eval_corpus = [item for sublist in eval_lines.values() for item in sublist]
    
    
    # Train corpus
    train_texts_and_labels = utils.convertToSubWordsSentencesAndLabels(train_lines, tokenizer=tokenizer, delimiter="£")
    train_dataset = trainer_functions.SentenceBoundaryDataset(train_texts_and_labels, tokenizer)

    # Dev corpus
    dev_texts_and_labels = utils.convertToSubWordsSentencesAndLabels(dev_lines, tokenizer=tokenizer, delimiter="£")
    dev_dataset = trainer_functions.SentenceBoundaryDataset(dev_texts_and_labels, tokenizer)

    if '/' in modelName:
        name_of_model = re.split('/', modelName)[1]
    else:
        name_of_model = modelName

    
    training_args = TrainingArguments(
        output_dir=f"results_{name_of_model}/epoch{num_train_epochs}_bs{batch_size}",
        num_train_epochs=num_train_epochs,
        logging_steps=logging_steps,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        evaluation_strategy="epoch",
        logging_strategy="epoch",
        dataloader_num_workers=8,
        dataloader_prefetch_factor=4,
        bf16=False,
        use_cpu=True if device == "cpu" else False,
        save_strategy="epoch",
        load_best_model_at_end=True
        # best model is evaluated on loss
    )

    # define the trainer : model, training args, datasets and the specific compute_metrics defined in functions file
    save_callback = SaveModelAndConfigCallback(save_dir=training_args.output_dir)
    
    trainer = CustomTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=dev_dataset,
        compute_metrics=trainer_functions.compute_metrics#,
        #callbacks=[save_callback]
    )
    # And then a global evaluation

    # fine-tune the model
    print("Starting training")
    trainer.train()
    print("End of training")

    
    # get the best model path
    best_model_path = trainer.state.best_model_checkpoint
    print(f"Evaluation.")

    # print the whole log_history with the compute metrics
    best_precision_step, best_step_metrics = utils.get_best_step(trainer.state.log_history)
    best_model_path = f"results_{name_of_model}/epoch{num_train_epochs}_bs{batch_size}/checkpoint-{best_precision_step}"
    config.to_json_file(json_file_path=f"{best_model_path}/config.json")
    # best_model_path = "results_bert-base-multilingual-cased/epoch1_bs264/checkpoint-1/"
    print(f"Best model path according to precision: {best_model_path}")
    print(f"Full metrics: {best_step_metrics}")
    # model.save_pretrained("results_bert-base-multilingual-cased/before_training/")
    
    # Ici il faut faire plusieurs tests différents.
    # best_model_path = "results_bert-base-multilingual-cased/before_training/"
    


    try:
        os.mkdir("tokenisation_training_results/")
    except FileExistsError:
        pass
    try:
        os.mkdir(f"tokenisation_training_results/results_{train_name}")
    except FileExistsError:
        pass

    os.rename(best_model_path, new_best_path)

    with open(f"{new_best_path}/model_name", "w") as model_name:
        model_name.write(modelName)


    with open(f"{new_best_path}/metrics.json", "w") as metrics:
        json.dump(best_step_metrics, metrics)

    

    # We perform evaluation by lang
    for lang, lines in eval_lines.items():
        eval_results = evaluation.run_eval(data=lines,
                                           model_path=new_best_path,
                                           tokenizer_name=tokenizer.name_or_path,
                                           verbose=False,
                                           lang=lang)

        with open(f"{new_best_path}/eval_{lang}.txt", "w") as evaluation_results:
            evaluation_results.write(eval_results)

    # Full dataset evaluation
    eval_results = evaluation.run_eval(data=full_eval_corpus,
                                       model_path=new_best_path,
                                       tokenizer_name=tokenizer.name_or_path,
                                       verbose=False)

    with open(f"{new_best_path}/eval.txt", "w") as evaluation_results:
        evaluation_results.write(eval_results)
    


    # We move the best state dir name to "best"
    #### CONTINUER ICI
    print(f"\n\nBest model can be found at : {new_best_path} ")
    print(
        f"You should remove the following directories by using `rm -r results_{name_of_model}/epoch{num_train_epochs}_bs{batch_size}/checkpoint-*`")

    # functions returns best model_path
    return new_best_path


# list of arguments to provide and application of the main function
if __name__ == '__main__':
    set_seed(42)

    parser = argparse.ArgumentParser()
    parser.add_argument("-m", "--model", default=None,
                        help="Base model to finetune.")
    parser.add_argument("-t", "--train_dataset", default="",
                        help="Path to train dataset.")
    parser.add_argument("-d", "--dev_dataset", default="",
                        help="Path to dev dataset.")
    parser.add_argument("-n", "--name", default="",
                        help="Training session name (will appear in created dir).")
    parser.add_argument("-e", "--eval_dataset", default="",
                        help="Path to eval dataset.")
    parser.add_argument("-dv", "--device", default="cpu",
                        help="Device to be used for training.")
    parser.add_argument("-ep", "--epochs", default=10,
                        help="Number of epochs to be realized.")
    parser.add_argument("-fm", "--freeze_metadata", default=False,
                        help="Whether to train with or without metadata embedding.")
    parser.add_argument("-b", "--batch_size", default=32,
                        help="Batch size.")
    parser.add_argument("-l", "--logging_steps", default=500)
    args = parser.parse_args()
    model = args.model
    train_dataset = args.train_dataset
    dev_dataset = args.dev_dataset
    eval_dataset = args.eval_dataset
    num_train_epochs = int(args.epochs)
    freeze_metadata = True if args.freeze_metadata == "True" else False
    batch_size = int(args.batch_size)
    logging_steps = int(args.logging_steps)
    device = args.device
    name = args.name
    datasets = {}
    training_trainer(model, datasets, num_train_epochs, batch_size, logging_steps, freeze_metadata=freeze_metadata, device=device, train_name=name)

