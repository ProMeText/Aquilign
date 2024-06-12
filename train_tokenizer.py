# -*- coding: utf-8 -*-
import sys
from transformers import BertTokenizer, Trainer, TrainingArguments, AutoModelForTokenClassification, set_seed
import aquilign.preproc.tok_trainer_functions as trainer_functions
import aquilign.preproc.eval as evaluation
import aquilign.preproc.utils as utils
import re
import os
import json
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
def training_trainer(modelName, train_dataset, dev_dataset, eval_dataset, num_train_epochs, batch_size, logging_steps, keep_punct=True, add_lang_metadata=True):
    model = AutoModelForTokenClassification.from_pretrained(modelName, num_labels=4)
    tokenizer = BertTokenizer.from_pretrained(modelName, max_length=10)
    train_lines = []
    dev_lines = []
    eval_lines = []
    train_dataset = glob.glob(f"data/tokenisation/*/*train.txt")
    for tdataset in train_dataset:
        lang = tdataset.split("/")[-2]
        with open(tdataset, "r") as train_file:
            current_train_lines = [(item.replace("\n", ""), lang) for item in train_file.readlines()]
            if keep_punct is False:
                current_train_lines = [(utils.remove_punctuation(line), lang) for line in current_train_lines]
            train_lines.extend(current_train_lines)

    dev_dataset = glob.glob(f"data/tokenisation/*/*dev.txt")
    for ddataset in dev_dataset:
        lang = ddataset.split("/")[-2]
        with open(ddataset, "r") as dev_file:
            current_dev_lines = [(item.replace("\n", ""), lang) for item in dev_file.readlines()]
            if keep_punct is False:
                current_dev_lines = [utils.remove_punctuation(line) for line in current_dev_lines]
            dev_lines.extend(current_dev_lines)

    
    # Train corpus
    train_texts_and_labels, tokenizer = utils.convertToSubWordsSentencesAndLabels(train_lines, tokenizer=tokenizer, delimiter="£", add_lang_metadata=add_lang_metadata)
    train_dataset = trainer_functions.SentenceBoundaryDataset(train_texts_and_labels, tokenizer)
    
    # Dev corpus
    dev_texts_and_labels, tokenizer = utils.convertToSubWordsSentencesAndLabels(dev_lines, tokenizer=tokenizer, delimiter="£", add_lang_metadata=add_lang_metadata)
    dev_dataset = trainer_functions.SentenceBoundaryDataset(dev_texts_and_labels, tokenizer)
    
    
    # We update the tokens embedding size, 
    # see https://huggingface.co/docs/transformers/main_classes/tokenizer#transformers.PreTrainedTokenizer.add_tokens
    model.resize_token_embeddings(len(tokenizer))

    
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
    
    
    # print the whole log_history with the compute metrics
    best_precision_step, best_step_metrics = utils.get_best_step(trainer.state.log_history)
    best_model_path = f"results_{name_of_model}/epoch{num_train_epochs}_bs{batch_size}/checkpoint-{best_precision_step}"
    
    # We save the tokenizer, in case it's been updated with new data
    tokenizer.save_pretrained(f"./results_{name_of_model}/epoch{num_train_epochs}_bs{batch_size}/checkpoint-{best_precision_step}/tokenizer")
    print(f"Best model path according to precision: {best_model_path}")
    print(f"Full metrics: {best_step_metrics}")
    
    

    eval_dataset = glob.glob(f"data/tokenisation/*/*eval.txt")
    for edataset in eval_dataset:
        lang = edataset.split("/")[-2]
        with open(edataset, "r") as eval_files:
            eval_lines = [(item.replace("\n", ""), lang) for item in eval_files.readlines()]
            if keep_punct is False:
                eval_lines = [utils.remove_punctuation(line) for line in eval_lines]
        eval_data_lang = edataset.split("/")[-2]
        print(f"\n---\nEvaluating model on {eval_data_lang}")
        eval_results = evaluation.run_eval(data=eval_lines,
                                           model_path=best_model_path,
                                           tokenizer_name=tokenizer.name_or_path,
                                           verbose=False,
                                           lang=eval_data_lang,
                                           add_lang_metadata=add_lang_metadata)
    
    
    
    

    # We move the best state dir name to "best"
    #### CONTINUER ICI
    new_best_path = f"results_{name_of_model}/epoch{num_train_epochs}_bs{batch_size}/best"
    try:
        os.rmdir(new_best_path)
    except FileNotFoundError:
        pass
    os.rename(best_model_path, new_best_path)
    
    with open(f"{new_best_path}/model_name", "w") as model_name:
        model_name.write(modelName)

    with open(f"{new_best_path}/eval.txt", "w") as evaluation_results:
        evaluation_results.write(eval_results)

    with open(f"{new_best_path}/metrics.json", "w") as metrics:
        json.dump(best_step_metrics, metrics)
        
    with open(f"{new_best_path}/model_info.txt", "w") as model_info:
        model_info.write(f"Lang metadata: {add_lang_metadata}")
    
    print(f"\n\nBest model can be found at : {new_best_path} ")
    print(f"You should remove the following directories by using `rm -r results_{name_of_model}/epoch{num_train_epochs}_bs{batch_size}/checkpoint-*`")

    # functions returns best model_path
    return new_best_path


# list of arguments to provide and application of the main function
if __name__ == '__main__':
    set_seed(42)
    model = sys.argv[1]
    train_dataset = sys.argv[2]
    dev_dataset = sys.argv[3]
    eval_dataset = sys.argv[4]
    num_train_epochs = int(sys.argv[5])
    batch_size = int(sys.argv[6])
    logging_steps = int(sys.argv[7])
    add_lang_metadata = True if sys.argv[8] == "True" else False

    training_trainer(model, train_dataset, dev_dataset, eval_dataset, num_train_epochs, batch_size, logging_steps, add_lang_metadata=add_lang_metadata)

