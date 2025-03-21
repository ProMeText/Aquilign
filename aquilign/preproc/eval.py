import aquilign.preproc.tok_trainer_functions as functions
import aquilign.preproc.syntactic_tokenization as SyntacticTok
import aquilign.preproc.create_train_data as FormatData
import aquilign.preproc.utils as utils
import sys
from transformers import BertTokenizer, AutoModelForTokenClassification, pipeline
import re
import torch
import numpy as np
import evaluate
from tabulate import tabulate
import unicodedata

def unalign_labels(human_to_bert, predicted_labels, splitted_text):
    realigned_list = []
    # itering on original text
    final_prediction = []
    for idx, value in enumerate(splitted_text):
        index = idx + 1
        predicted = human_to_bert[index]
        # if no mismatch, copy the label
        if len(predicted) == 1:
            correct_label = predicted_labels[predicted[0]]
        # mismatch
        else:
            ###
            correct_label = [predicted_labels[predicted[n]] for n in range(len(predicted))]
            # Dans ce cas on regarde s'il y a 1 dans n'importe quelle position des rangs correspondants:
            # on considère que BERT ne propose qu'une tokénisation plus importante que nous
            if any([n == 1 for n in correct_label]):
                correct_label = 1
            elif any([n == 2 for n in correct_label]):
                correct_label = 0
            else:
                correct_label = 0

        final_prediction.append(correct_label)

    assert len(final_prediction) == len(splitted_text), "List mismatch"
    tokenized_sentence = " ".join(
        [element if final_prediction[index] != 1 else f"\n{element}" for index, element in enumerate(splitted_text)])
    return final_prediction


def get_labels_from_preds(preds):
    bert_labels = []
    for pred in preds[-1]:
        label = [idx for idx, value in enumerate(pred) if value == max(pred)][0]
        bert_labels.append(label)
    return bert_labels


def pad_list(input_list:list, tgt_length:int) -> list:
    out_padded_list = []
    for item in input_list:
        pad_num = tgt_length - len(item)
        padding = [2 for item in range(pad_num)]
        padded_list = item
        padded_list.extend(padding)
        out_padded_list.append(padded_list)
    return out_padded_list

def get_metrics(preds, tgts):
    """
    This function produces the metrics for evaluating the model at the end of training
    """
    accuracies = evaluate.load("accuracy")
    all_other_metrics = evaluate.combine(["recall", "precision", "f1"])
    all_accs, all_recall, all_precision, all_f1 = [], [], [], []
    max_preds_length = max(len(pred) for pred in preds)
    max_tgts_length = max(len(tgt) for tgt in tgts)
    assert max_tgts_length == max_tgts_length, "Issue with max length"
    padded_preds = pad_list(preds, max_preds_length)
    padded_tgts = pad_list(tgts, max_preds_length)
    for preds, refs in zip(padded_preds, padded_tgts):
        accuracies.add_batch(references=refs, predictions=preds)
        all_other_metrics.add_batch(references=refs, predictions=preds)
    accuracy = accuracies.compute()['accuracy']
    metrics = all_other_metrics.compute(average=None)
    recall = metrics['recall'].tolist()
    precision = metrics['precision'].tolist()
    f1 = metrics['f1'].tolist()
    return accuracy, precision, recall ,f1
    
    
# correspondences between our labels and labels from the BERT-tok
def get_correspondence(sent, tokenizer, delimiter):
    out = {}
    # First token is CLS
    tokenized_index =  1
    out[0] = (0,)
    for index, word in enumerate(utils.tokenize_words(sent, delimiter)):
        tokenized_word = tokenizer.tokenize(word)
        out[index + 1] = tuple(item for item in range(tokenized_index, tokenized_index + len(tokenized_word)))
        tokenized_index += len(tokenized_word)
    human_split_to_bert = out
    bert_split_to_human_split = {value: key for key, value in human_split_to_bert.items()}
    return human_split_to_bert, bert_split_to_human_split

def unicode_normalise(string:str) -> str:
    return unicodedata.normalize("NFC", string)

def run_eval(data:list|str, model_path, tokenizer_name, verbose=False, delimiter="£", standalone=False, remove_punctuation=False, lang=None):
    if standalone:
        with open(data, "r") as input_file:
            corpus_as_list = [unicode_normalise(item.replace("\n", "")) for item in input_file.readlines()]
        lang = data.split("/")[-2]
    else:
        corpus_as_list = [unicode_normalise(item) for item in data]
    
    if remove_punctuation:
        corpus_as_list = [utils.remove_punctuation(item) for item in corpus_as_list]
    
    all_preds, all_tgts = [], []
    tokenizer = BertTokenizer.from_pretrained(tokenizer_name, max_length=10)
    new_model = AutoModelForTokenClassification.from_pretrained(model_path, num_labels=3)
    # get the path of the default tokenizer
    texts, labels, tokenized_text = utils.convertToWordsSentencesAndLabels(corpus_as_list)
    assert len(texts) == len(labels),  "Lists mismatch"
    
    print("Performing syntactic tokenization evaluation")
    # First, regexp evaluation
    syntactic_preds, all_syntactic_gt = [], []
    for idx, (example, label, tokens) in enumerate(zip(texts, labels, tokenized_text)):
        if verbose:
            print("---\nSYNTtok New example")
            print(f"Example:   {example}")
        tokenized_text = SyntacticTok.syntactic_tokenization(input_file=None, 
                                                        standalone=False, 
                                                        text=example,
                                                        use_punctuation=False,
                                                        lang=lang)
        formatted_preds = [f" {delimiter}".join(tokenized_text)]
        preds_to_labels = utils.convertToWordsSentencesAndLabels(formatted_preds)
        if verbose:
            print(f"Tokenized text: {tokenized_text}")
            print(f"Formatted: {formatted_preds}")
            print(f"As labels: {preds_to_labels}")
            print(list(zip(preds_to_labels[1][0], preds_to_labels[2][0])))
        
        # Si la fonction words_to_labels ne renvoie que des listes vides, c'est que la tokénisation n'a rien donné.
        if preds_to_labels == ([], [], []):
            predicted = [0 for item in label]
        else:
            predicted = preds_to_labels[1][0]
            
        syntactic_preds.append(predicted)
        all_syntactic_gt.append(label)
        syntactic_preds.append(predicted)
        all_syntactic_gt.append(label)
        if verbose:
            print(f"Predicted:    {predicted}")
            print(f"Ground Truth: {label}")
            print(f"Example length: {len(example.split())}")
            print(f"Preds length: {len(predicted)}")
            print(f"GT length: {len(label)}")
            print(f"Orig GT: {corpus_as_list[idx]}")

        assert len(predicted) == len(label), f"Length mismatch, please check the regular expressions don't split any word:\n" \
                                             f"{example}\n" \
                                             f"(label: {len(label)} and predicted: {len(predicted)})" \
                                             f"Labels: {label}" \
                                             f"Preds: {predicted}"
            
    synt_results = get_metrics(syntactic_preds, all_syntactic_gt)
    print(synt_results)
    
    
    # Second, model evaluation
    print("Performing bert-based tokenization evaluation")
    gt_toks_and_labels = utils.convertToSubWordsSentencesAndLabels(corpus_as_list, tokenizer=tokenizer, delimiter=delimiter)
    for txt_example, gt in zip(corpus_as_list, gt_toks_and_labels):
        # We get only the text
        example = txt_example.replace(delimiter, "")
        splitted_example = utils.tokenize_words(example, delimiter)
        # BERT-tok
        enco_nt_tok = tokenizer.encode(example, truncation=True, padding=True, return_tensors="pt")
        # get the predictions from the model
        predictions = new_model(enco_nt_tok)
        
        preds = predictions[0]
        # apply the functions
        bert_labels = get_labels_from_preds(preds)
        
        # On crée la table de correspondance entre les words et les subwords
        human_to_bert, _ = get_correspondence(example, tokenizer, delimiter)
        unaligned_preds = unalign_labels(human_to_bert, bert_labels, splitted_example)
        unaligned_tgts = unalign_labels(human_to_bert, gt['labels'].tolist(), splitted_example)
        
        # On remet la première et la dernière prédiction qui correspond au [CLS] et [SEP] et n'est pas prise en compte dans le réalignement
        unaligned_preds.insert(0, bert_labels[0])
        unaligned_tgts.insert(0, gt['labels'].tolist()[0])
        unaligned_preds.append(bert_labels[-1])
        unaligned_tgts.append(gt['labels'].tolist()[:len(bert_labels)][-1])
        
        assert len(unaligned_preds) == len(unaligned_tgts), f"Target and Preds mismatch, please check data: " \
                                                       f"\n{unaligned_preds}" \
                                                       f"\n{unaligned_tgts}"
        all_preds.append(unaligned_preds)
        all_tgts.append(unaligned_tgts)
        if verbose:
            print(f"---\nNew example: {example}")
            print(f"Example lenght: {len(splitted_example)}")
            print(f"Bert Tokenized: {enco_nt_tok.tolist()}")
            print(f"Tokens: {tokenizer.convert_ids_to_tokens(ids=enco_nt_tok.tolist()[0])}")
            print(f"Preds Labels length:  {len(bert_labels)}")
            print(f"Tokens length: {len(enco_nt_tok.tolist()[0])}")
            print(f"Zip: {list(zip(tokenizer.convert_ids_to_tokens(ids=enco_nt_tok.tolist()[0]), bert_labels))}")
            print(f"Bert labels (subwords):            {bert_labels}")
            print(f"Truncated ground truth (subwords): {gt['labels'].tolist()[:len(bert_labels)]}")
            print("Unaligning labels")
            print(f"First bert label: {bert_labels[0]}")
            print(f"Last bert label: {bert_labels[-1]}")
            print(f"Predictions (words):               {unaligned_preds}")
            print(f"Targets (words):                   {unaligned_tgts}")
            print(f"Length: {len(unaligned_preds)}")
            assert len(unaligned_preds) - len(splitted_example) == 2, "Something went wrong during words/subwords alignment"
    bert_results = get_metrics(all_preds, all_tgts)
    
    zipped_results = list(zip(['Accuracy', 'Precision', 'Recall', 'F1-score'], synt_results, bert_results))
    print(tabulate(zipped_results, headers=['', 'Synt (None, Delim.)', 'Bert (None, Delim., Pad.)'], tablefmt='orgtbl'))
    return tabulate(zipped_results, headers=['', 'Synt (None, Delim.)', 'Bert (None, Delim., Pad.)'], tablefmt='orgtbl')
        



if __name__ == '__main__':
    file_to_test = sys.argv[1]
    model_path = sys.argv[2]
    tokenizer_name = sys.argv[3]
    run_eval(file_to_test, model_path, tokenizer_name, standalone=True)