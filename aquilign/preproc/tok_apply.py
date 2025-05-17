# -*- coding: utf-8 -*-

import sys
import os
from os.path import join
from transformers import BertTokenizer, AutoModelForTokenClassification
import re
import langid
import tqdm


## script for applying the tokenisation to text
## it produces .txt files which has been tokenized ; each element of tokenisation is marked by a breakline

## usage : python tok_apply.py model_name tokenizer_name text_to_tokenize.txt tokens_per_example output_dir
## where :
# model_name is the model for the tokenization ie the token classificator which has been trained and got good results enough
# tokenizer_name is the name of the BERT tokenizer, equal to the default BERT model language (important because it changes depends on the language)
# text_to_tokenize.txt is the path to the .txt file we want to tokenize
# tokens_per_example is the number of tokens (words and pc) we want to send to the classifier (max BERT length is 512) ; recommended : use the number which was used for the training
# output_dir is the name of the output directory, where the produced files will be saved


### functions


def remove_punctuation(text: str):
    punct = re.compile(r"[\.,;—:\?!’'«»“/\-]")
    cleaned_text = re.sub(punct, "", text)
    return cleaned_text


# tokenize text (BERT as a max length of 512) ; recommended : get the same length as for the training
def tokenize(text, tokens_per_example):
    words = text.split(" ")
    return [' '.join(words[i:i + tokens_per_example]) for i in range(0, len(words), tokens_per_example)]


# get the labels
def get_labels_from_preds(preds):
    bert_labels = []
    for pred in preds[-1]:
        label = [idx for idx, value in enumerate(pred) if value == max(pred)][0]
        bert_labels.append(label)
    return bert_labels


# correspondences between our labels and labels from the BERT-tok
def get_correspondence(sent, tokenizer, verbose=False):
    out = {}
    tokenized_index = 0
    for index, word in enumerate(sent):
        # print(tokenizer.tokenize(word))
        tokenized_word = tokenizer.tokenize(word)
        if verbose:
            print(tokenized_word)
        out[index] = tuple(item for item in range(tokenized_index, tokenized_index + len(tokenized_word)))
        tokenized_index += len(tokenized_word)
    human_split_to_bert = out
    bert_split_to_human_split = {value: key for key, value in human_split_to_bert.items()}
    return human_split_to_bert, bert_split_to_human_split


def unalign_labels(human_to_bert, predicted_labels, splitted_text, verbose=False):
    predicted_labels = predicted_labels[1:-1]
    if verbose:
        print(f"Prediction: {predicted_labels}")
        print(human_to_bert)
        print(splitted_text)
    realigned_list = []

    # itering on original text
    final_prediction = []
    for index, value in enumerate(splitted_text):
        predicted = human_to_bert[index]
        # if no mismatch, copy the label
        if len(predicted) == 1:
            correct_label = predicted_labels[predicted[0]]
            if verbose:
                print(f"Position {index}")
                print(predicted_labels)
                print(predicted[0])
                print(correct_label)
        # mismatch
        else:
            correct_label = [predicted_labels[predicted[n]] for n in range(len(predicted))]
            if verbose:
                print(f"predicted labels mismatch :{predicted_labels}")
                print(f"len predicted mismatch {len(predicted)}")
                print(f"Corresponding labels in prediction: {correct_label}")
            # Dans ce cas on regarde s'il y a 1 dans n'importe quelle position des rangs correspondants:
            # on considère que BERT ne propose qu'une tokénisation plus importante que nous
            if any([n == 1 for n in correct_label]):
                correct_label = 1
        final_prediction.append(correct_label)

    assert len(final_prediction) == len(splitted_text), "List mismatch"

    tokenized_sentence = " ".join(
        [element if final_prediction[index] != 1 else f"\n{element}" for index, element in enumerate(splitted_text)])
    if verbose:
        print(f'final prediction {final_prediction}')
        print(tokenized_sentence)
    return tokenized_sentence


def tokenize_text(input_file: str,
                  model_path=None,
                  tokenizer_name=None,
                  remove_punct=False,
                  tok_models: dict = None,
                  corpus_limit=None,
                  output_dir=None,
                  tokens_per_example=None,
                  device="cpu",
                  verbose=False,
                  lang=None):
    """
    Performs tokenization with given model, tokenizer on given file
    """

    with open(input_file) as f:
        textL = f.read().splitlines()
    localText = " ".join(str(element) for element in textL)
    if corpus_limit:
        localText = localText[:round(len(localText) * corpus_limit)]
    if remove_punct:
        localText = remove_punctuation(localText)

    if not lang:
        codelang, _ = langid.classify(localText[:300])
        # Il ne reconnaît pas toujours le castillan
        if codelang == "an" or codelang == "oc" or codelang == "pt" or codelang == "gl":
            codelang = "es"
        if codelang == "eo" or codelang == "ht":
            codelang = "fr"
        if codelang == "jv":
            codelang = "it"
        print(f"Detected lang: {codelang}")
    else:

        codelang = lang

    # get the path of the model
    if model_path:
        pass
    else:
        model_path = tok_models[codelang]["model"]
        tokens_per_example = tok_models[codelang]["tokens_per_example"]
        tokenizer_name = tok_models[codelang]["tokenizer"]

    print(f"Using {model_path} model and {tokenizer_name} tokenizer.")
    new_model = AutoModelForTokenClassification.from_pretrained(model_path, num_labels=3)
    # get the path of the default tokenizer
    tokenizer = BertTokenizer.from_pretrained(tokenizer_name, max_length=tokens_per_example)
    new_model.to(device)

    # get the number of tokens per fragment to tokenize
    if not tokens_per_example:
        tokens_per_example = tok_models[codelang]["tokens_per_example"]
    # split the full input text as slices
    text = tokenize(localText, tokens_per_example)
    # prepare the data
    restruct = []
    # apply the tok process on each slice of text
    id_to_token = {i: token for i, token in enumerate(tokenizer.get_vocab().keys())}
    for i in tqdm.tqdm(text):
        # BERT-tok
        enco_nt_tok = tokenizer.encode(i, truncation=True, padding=True, return_tensors="pt")
        enco_nt_tok = enco_nt_tok.to(device)
        # get the predictions from the model
        predictions = new_model(enco_nt_tok)
        preds = predictions[0]
        # apply the functions
        bert_labels = get_labels_from_preds(preds)
        human_to_bert, bert_to_human = get_correspondence(i.split(), tokenizer)
        new_labels = unalign_labels(human_to_bert=human_to_bert, predicted_labels=bert_labels, splitted_text=i.split())
        tokenized = new_labels.split("\n")
        if verbose:
            print(i)
            print(new_labels)
            print(tokenized)

        # Gestion du premier token.
        try:
            if tokenized[0] == "":
                restruct.extend(tokenized[1:])
            else:
                last_token = restruct[-1]
                restruct[-1] = f"{last_token} {tokenized[0]}"
                restruct.extend(tokenized[1:])
        # Pour le premier token
        except IndexError:
            if tokenized[0] == "":
                restruct.extend(tokenized[1:])
            else:
                restruct.extend(tokenized)

    # On teste la non perte de tokens
    input_text_length = len(localText.split())
    output_text_length = len(" ".join(restruct).split())

    assert input_text_length == input_text_length, "Length of input text and tokenized text mismatch, something went wrong: " \
                                                   f"Input: {input_text_length}, output: {output_text_length}"
    print("No tokens were lost during the process.")

    # prepare the name of the output file
    if '/' in input_file:
        filename_corr = input_file.rpartition('/')[-1].split('.')[0]
    else:
        filename_corr = input_file.split('.')[0]

    output_file = join(output_dir, f'{filename_corr}-tok.txt')

    # create or no the directory
    # TODO: corriger ça
    try:
        os.mkdir(output_dir)
    except OSError as exception:
        pass

    # write the file
    with open(f"result_dir/{output_file}", "w") as text_file:
        text_file.write("\n".join(restruct))
        print(f"Saving to {output_file}")
    return restruct


###
if __name__ == '__main__':
    model_path = sys.argv[1]
    tokenizer_name = sys.argv[2]
    remove_punct = False
    input_file = sys.argv[3]
    example_length = int(sys.argv[4])
    output_dir = sys.argv[5]
    device = sys.argv[6]

    tokenize_text(model_path=model_path,
                  tokenizer_name=tokenizer_name,
                  remove_punct=remove_punct,
                  input_file=input_file,
                  tokens_per_example=example_length,
                  device=device,
                  output_dir=output_dir)
