

# -*- coding: utf-8 -*-

import sys
import os
from os.path import join, isfile
from os import listdir
from transformers import BertTokenizer, AutoModelForTokenClassification
import argparse

## script for applying the tokenisation to text
## it produces .txt files which has been tokenized ; each element of tokenisation is marked by a breakline

## usage : python tok_apply.py -type=folder|file model_name tokenizer_name text_to_tokenize.txt num output_dir
## where :
# -type : indicates if we want to tokenize one file or a folder with several files (authorized values : folder or file)
# model_name is the model for the tokenization ie the token classificator which has been trained and got good results enough
# tokenizer_name is the name of the BERT tokenizer, equal to the default BERT model language (important because it changes depends on the language)
# text_to_tokenize.txt is the path to the .txt file we want to tokenize
# num is the number of tokens (words and pc) we want to send to the classifier (max BERT length is 512) ; recommended : use the number which was used for the training
# output_dir is the name of the output directory, where the produced files will be saved



### functions

# tokenize text (BERT as a max length of 512) ; recommended : get the same length as for the training
def tokenize(text,num):
    words = text.split(" ")
    return [' '.join(words[i:i+num]) for i in range(0, len(words), num)]


#get the labels
def get_labels_from_preds(preds):
    bert_labels = []
    for pred in preds[-1]:
        label = [idx for idx, value in enumerate(pred) if value == max(pred)][0]
        bert_labels.append(label)
    return bert_labels

# correspondences between our labels and labels from the BERT-tok
def get_correspondence(sent, tokenizer):
    out = {}
    tokenized_index = 0
    for index, word in enumerate(sent):
        # print(tokenizer.tokenize(word))
        tokenized_word = tokenizer.tokenize(word)
        print(tokenized_word)
        out[index] = tuple(item for item in range(tokenized_index, tokenized_index + len(tokenized_word)))
        tokenized_index += len(tokenized_word)
    human_split_to_bert = out
    bert_split_to_human_split = {value: key for key, value in human_split_to_bert.items()}
    return human_split_to_bert, bert_split_to_human_split

def unalign_labels(bert_to_human, predicted_labels, splitted_text):
    predicted_labels = predicted_labels[1:-1]
    print(f"Prediction: {predicted_labels}")
    print(human_to_bert)
    ###
    # itering on original text
    final_prediction = []
    for index, value in enumerate(splitted_text):
        print(f"Position {index}")
        predicted = human_to_bert[index]

        # if no mismatch, copy the label
        if len(predicted) == 1:
            print(predicted_labels)
            print(predicted[0])
            correct_label = predicted_labels[predicted[0]]
            print(correct_label)
        # mismatch
        else:
            ###
            print(f"predicted labels mismatch :{predicted_labels}")
            print(f"len predicted mismatch {len(predicted)}")
            correct_label = [predicted_labels[predicted[n]] for n in range(len(predicted))]
            print(f"Corresponding labels in prediction: {correct_label}")
            # Dans ce cas on regarde s'il y a 1 dans n'importe quelle position des rangs correspondants:
            # on considère que BERT ne propose qu'une tokénisation plus importante que nous
            if any([n == 1 for n in correct_label]):
                correct_label = 1
        final_prediction.append(correct_label)

    assert len(final_prediction) == len(splitted_text), "List mismatch"
    print(f'final prediction {final_prediction}')

    tokenized_sentence = " ".join(
        [element if final_prediction[index] != 1 else f"\n{element}" for index, element in enumerate(splitted_text)])
    print(tokenized_sentence)
    return tokenized_sentence

# add the arguments
parser = argparse.ArgumentParser()
parser.add_argument('-type', '--type', help='Indicate here if you want to tokenize a file or a folder. Authorized values are : file, folder', type=str)
parser.add_argument('model_path', help='Path of the model we want to use', type=str)
parser.add_argument('tok_name', help='Name of the BERT tokenizer (different in function of language)', type=str)
parser.add_argument('input', help='Path of the input, folder or file', type=str)
parser.add_argument('num', help='Length of the slices of the texts', type=str)
parser.add_argument('output_dir', help='Name of the output directory', type=str)
args = parser.parse_args()

# load the model we trained for tokenization
new_path = args.model_path
new_model = AutoModelForTokenClassification.from_pretrained(new_path, num_labels=3)
new_model

# get the path of the default tokenizer
tokenizer_name = args.tok_name
tokenizer = BertTokenizer.from_pretrained(tokenizer_name, max_length=10)


#get the number of tokens per fragment to tokenize
num = int(args.num)
#name of output dir
output_dir = args.output_dir
#name of input (folder or file)
input = args.input

#creation of output dir
try:
    os.mkdir(output_dir)
except OSError as exception:
    pass


### if type is a folder
if args.type == 'folder' :
    directory = os.fsencode(input)
    #iterate on the files in the folder
    for filename in listdir(directory):
        full_path = join(directory, filename)
        if isfile(full_path):
            with open(full_path) as f:
                textL = f.read().splitlines()
                localText = " ".join(str(element) for element in textL)
                #cut the text in slices
                text = tokenize(localText, num)
                print(text)
                restruct = ""
                #process on each slice of text
                for i in text :
                    #BERT-tok
                    enco_nt_tok = tokenizer.encode(i, truncation=True, padding=True, return_tensors="pt")
                    #predictions of the model
                    predictions = new_model(enco_nt_tok)
                    preds = predictions[0]
                    #apply all the functions on the slices
                    bert_labels = get_labels_from_preds(preds)
                    human_to_bert, bert_to_human = get_correspondence(i.split(), tokenizer)
                    new_labels = unalign_labels(bert_to_human, bert_labels, i.split())
                    #append the string with the new tokenized value
                    new_labels_sp = new_labels + ' '
                    restruct += new_labels_sp
                    print(restruct)
                #create the final file
                final_filename = filename.decode("utf-8").split('.')[0]
                output_file = join(output_dir, f'{final_filename}-tok.txt')
                with open(output_file, "w") as text_file:
                    text_file.write(restruct)

### if type is a file
elif args.type == 'file':
    #get the file
    input_file = input
    with open(input_file) as f:
        textL = f.read().splitlines()
        localText = " ".join(str(element) for element in textL)
    #cut the text in slices
    text = tokenize(localText, num)
    #prepare the data
    restruct = ""
    # apply the tok process on each slice of text
    for i in text:
        # BERT-tok
        enco_nt_tok = tokenizer.encode(i, truncation=True, padding=True, return_tensors="pt")
        #get the predictions from the model
        predictions = new_model(enco_nt_tok)
        preds = predictions[0]
        #apply the functions
        bert_labels = get_labels_from_preds(preds)
        human_to_bert, bert_to_human = get_correspondence(i.split(), tokenizer)
        new_labels = unalign_labels(bert_to_human, bert_labels, i.split())
        #append the string with the new tokenized value
        new_labels_sp = new_labels + ' '
        restruct += new_labels_sp
        print(restruct)

    # prepare the name of the output file
    if '/' in input_file:
        filename_corr = input_file.rpartition('/')[-1].split('.')[0]
    else:
        filename_corr = input_file.split('.')[0]
    output_file = join(output_dir, f'{filename_corr}-tok.txt')
    # write the file
    with open(output_file, "w") as text_file:
        text_file.write(restruct)

else:
        "You need to choose folder or file in the -type argument"


