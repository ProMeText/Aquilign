# -*- coding: utf-8 -*-
import sys
from transformers import BertTokenizer, Trainer, TrainingArguments, AutoModelForTokenClassification
import aquilign.preproc.tok_trainer_functions as trainer_functions
import aquilign.preproc.eval as evaluation
import aquilign.preproc.utils as utils
import re
import shutil
import glob
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

# function which produces the train, which first gets texts, transforms them into tokens and labels, then trains model with the specific given arguments
def training_trainer(modelName, train_dataset, dev_dataset, eval_dataset, num_train_epochs, batch_size, logging_steps, keep_punct=False):
    model = AutoModelForTokenClassification.from_pretrained(modelName, num_labels=3)
    tokenizer = BertTokenizer.from_pretrained(modelName, max_length=10)
    
    with open(train_dataset, "r") as train_file:
        train_lines = [item.replace("\n", "") for item in train_file.readlines()]
        if keep_punct is False:
            train_lines = [utils.remove_punctuation(line) for line in train_lines]
        
    with open(dev_dataset, "r") as dev_file:
        dev_lines = [item.replace("\n", "") for item in dev_file.readlines()]
        if keep_punct is False:
            dev_lines = [utils.remove_punctuation(line) for line in dev_lines]
        
    with open(eval_dataset, "r") as eval_files:
        eval_lines = [item.replace("\n", "") for item in eval_files.readlines()]
        if keep_punct is False:
            eval_lines = [utils.remove_punctuation(line) for line in eval_lines]
    
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

    # training arguments
    # num train epochs, logging_steps and batch_size should be provided
    # evaluation is done by epoch and the best model of each one is stored in a folder "results_+name"
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
        bf16=True,
        use_cpu=False,
        save_strategy="epoch",
        load_best_model_at_end=True
        # best model is evaluated on loss
    )

    # define the trainer : model, training args, datasets and the specific compute_metrics defined in functions file
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=dev_dataset,
        compute_metrics=trainer_functions.compute_metrics
    )

    # fine-tune the model
    print("Starting training")
    trainer.train()
    print("End of training")

    # get the best model path
    best_model_path = trainer.state.best_model_checkpoint
    print(f"Evaluation.")
    
    evaluation.run_eval(file=eval_lines, model_path=best_model_path, tokenizer_name=tokenizer.name_or_path, verbose=False)
    
    # print the whole log_history with the compute metrics
    print("\nBest model is evaluated on the loss results. Here is the log history with the performances of the models :")
    print(trainer.state.log_history)

    # We move the best state dir name to "best"
    new_best_path = f"results_{name_of_model}/epoch{num_train_epochs}_bs{batch_size}/best"
    print(f"\n\nBest model can be found at : {new_best_path} ")
    shutil.move(best_model_path, new_best_path)
    print(f"You should remove the following directories by using `rm -r results_{name_of_model}/epoch{num_train_epochs}_bs{batch_size}/checkpoint-*`")

    # functions returns best model_path
    return new_best_path


# list of arguments to provide and application of the main function
if __name__ == '__main__':
    model = sys.argv[1]
    train_dataset = sys.argv[2]
    dev_dataset = sys.argv[3]
    eval_dataset = sys.argv[4]
    num_train_epochs = int(sys.argv[5])
    batch_size = int(sys.argv[6])
    logging_steps = int(sys.argv[7])

    training_trainer(model, train_dataset, dev_dataset, eval_dataset, num_train_epochs, batch_size, logging_steps)

