# -*- coding: utf-8 -*-
import re
import torch
import evaluate
import numpy as np
import tqdm 






# function to get the index of the tokens after BERT tokenization
def get_index_correspondence(sent, tokenizer):
    correspondence = [(0,0)]
    for word in sent:
        (raw_end, expand_end) = correspondence[-1]
        tokenized_word = tokenizer.tokenize(word)
        correspondence.append((raw_end+1, expand_end+len(tokenized_word)))
    return correspondence


def align_labels(corresp, orig_labels, text):
# function to align labels between the tokens in input and the tokenized tokens
    new_labels = [0 for r in range(corresp[-1][1])]
    for index, label in enumerate(orig_labels):
        # label which is interesting : 1
        if label == 1:
            try:
                if len(new_labels) == corresp[index][1]:
                    new_labels[(corresp[index][1]) - 1] = 1
            except IndexError:
                print(new_labels)
                print("Error.")
                exit(0)
            else:
                try:
                    new_labels[(corresp[index][1])] = 1
                except IndexError:
                    print(f"Error with example:\n {text}.\n"
                          f"Exiting.")
        else:
            pass
    # for special tokens (automatically added by BERT tokenizer), value of 2
    new_labels.insert(0, 2)
    new_labels.append(2)
    return new_labels


# function who gets the max length of tokenized text, used then in the class SentenceBoundaryDataset
def get_token_max_length(train_texts, tokenizer):
    lengths_list = []
    for text in train_texts:
        tok_text = tokenizer(text, return_tensors='pt')
        # get the length for every tok text
        tensor_length = (tok_text['input_ids'].squeeze())
        length = tensor_length.shape[0]
        lengths_list.append(length)
    # get the max value from the list
    max_length = max(lengths_list)
    return max_length


# dataset class which fits the requirements
class SentenceBoundaryDataset(torch.utils.data.Dataset):
    def __init__(self, texts_and_labels, tokenizer):
        self.texts_and_labels = texts_and_labels

    def __len__(self):
        return len(self.texts_and_labels)

    def __getitem__(self, idx):
        # get the max length of the training set in order to have the good feature to put in tokenizer
        # current text (one line, ie 12 tokens [before automatic BERT tokenization])
        return self.texts_and_labels[idx]


def compute_metrics(eval_pred):
    print("Starting eval")
    # load the metrics we want to evaluate
    metric1 = evaluate.load("accuracy")
    metric2 = evaluate.load("recall")
    metric3 = evaluate.load("precision")
    metric4 = evaluate.load("f1")

    predictions, labels = eval_pred
    # get the label predictions
    predictions = np.argmax(predictions, axis=2)

    # get the right format
    predictions = np.array(predictions, dtype='int32').flatten()
    labels = np.array(labels, dtype='int32').flatten()

    # automatically, value of -100 are produce ; we haven't understood why but we change them to 0. If not, it will give poor results
    ###
    labels = [0 if x == -100 else x for x in labels]
    ###
    # print(predictions)
    # print(labels)

    acc = metric1.compute(predictions=predictions, references=labels)
    recall = metric2.compute(predictions=predictions, references=labels, average=None)
    recall_l = []
    [recall_l.extend(v) for k, v in recall.items()]
    precision = metric3.compute(predictions=predictions, references=labels, average=None)
    precision_l = []
    [precision_l.extend(v) for k, v in precision.items()]
    f1 = metric4.compute(predictions=predictions, references=labels, average=None)
    f1_l = []
    [f1_l.extend(v) for k, v in f1.items()]

    print("Eval finished")
    return {"accurracy": acc, "recall": recall_l, "precision": precision_l, "f1": f1_l}