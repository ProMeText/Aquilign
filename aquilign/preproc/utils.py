import re
import aquilign.preproc.tok_trainer_functions as functions
import torch

def tokenize(text,num):
    words = text.split(" ")
    return [' '.join(words[i:i+num]) for i in range(0, len(words), num)]


def get_best_precision(results):
    """
    This function gets the best precision of label 1 (= delimiter) given the results of the trainer
    """
    result_dict = {}
    for result in results:
        try:
            result_dict[result['step']] = {**result_dict[result['step']], **result}
        except KeyError:
            result_dict[result['step']] = result

    all_precisions = {}
    for key, value in result_dict.items():
        precision = value['eval_precision'][1]
        all_precisions[key] = precision

    best_step = next(step for step, precision in all_precisions.items() if precision == max(all_precisions.values()))
    print(f"Best step according to precision: {best_step}")
    return best_step, result_dict[best_step]

def remove_punctuation(text:str):
    punct = re.compile(r"[\.,;—:\?!’'«»“/\-]")
    cleaned_text = re.sub(punct, "", text)
    return cleaned_text
    


def tokenize_words(sentence:str) -> list:
    """
    Cette fonction tokénise une phrase selon un certain nombre de marqueurs
    """
    words_delimiters = re.compile(r"[\.,;—:\?!’'«»“/\-]|[^\.,;—:\?!’'«»“/\-\s]+")
    sentenceAsList = re.findall(words_delimiters, sentence)
    return sentenceAsList


def convertToWordsSentencesAndLabels(corpus:list, delimiter="£") -> (list, list):
    """
    This function take a corpus as a list of examples and returns the masks for each token as words
    """

    sentencesList = []
    sentencesAsLabels = []
    for text in corpus:
        sentenceAsList = tokenize_words(text)
        masks = []
        for token in sentenceAsList:
            if delimiter in token:
                masks.append(1)
            else:
                masks.append(0)
        sentencesAsLabels.append(masks)
        sentence = text.replace(delimiter, "")
        sentencesList.append(sentence)
    return sentencesList, sentencesAsLabels


# function to convert text in input as tokens and labels (if label is identified in the file, gives 1, in other cases, 0)
def convertToSubWordsSentencesAndLabels(corpus, tokenizer, delimiter="£",  verbose=False):
    """
    This function takes a corpus and returns the tokenized corpus as subwords with their labels.
    """
    if verbose:
        print("Converting to sentences and labels")
    sentencesList = []
    sentencesAsLabels = []
    for text, lang in corpus:
        sentenceAsList = tokenize_words(text)
        # We start with 2 that represents the lang metadata
        masks = []
        for token in sentenceAsList:
            if delimiter in token:
                masks.append(1)
            else:
                masks.append(0)
        sentencesAsLabels.append(masks)
        sentence = text.replace(delimiter, "")
        sentencesList.append((sentence, lang))
    num_max_length = functions.get_token_max_length(sentencesList, tokenizer) + 1
    out_toks_and_labels = []
    # langs = "es it fr"
    # [4, 1058, 4290, 2123, 5]
    lang_mapping = {"es": 1058, "it": 4290, "fr": 2123}
    for (text, lang), labels in zip(sentencesList, sentencesAsLabels):
        toks = tokenizer(text, padding="max_length", max_length=num_max_length, truncation=False,
                         return_tensors="pt")
        # get the text with the similar splits as for the creation of the data
        tokens = tokenize_words(text)
        # get the index correspondences between text and tok text
        corresp = functions.get_index_correspondence(tokens, tokenizer)
        # aligning the label
        new_labels = functions.align_labels(corresp, labels)
        #print(new_labels)
        #print(f"New labels: {len(new_labels)}")
        #print([tokenizer.convert_ids_to_tokens(t) for t in tokenizer.encode(text)])
        # get the length of the tensor
        
        # We add the metadata: CONTINUE HEERE
        sq = (toks['input_ids'].squeeze())
        
        attention_mask:list = toks['attention_mask'].squeeze().tolist()
        attention_mask.insert(0, 1)
        #print(f"Attention: {len(attention_mask)}")
        attention_mask = torch.tensor(attention_mask)
        
        sq = sq.tolist()
        sq.insert(1, lang_mapping[lang])
        #print(sq)
        #print(f"SQ: {len(sq)}")
        sq = torch.tensor(sq)
        sq_as_list = sq.tolist()
        # print(list(zip([tokenizer.convert_ids_to_tokens(t) for t in tokenizer.encode(text)], [t for t in tokenizer.encode(text)], [item for item in new_labels])))
        
        ### insert 2 for in the new_labels in order to get tensors with the same size !
        if len(sq) == len(new_labels):
            pass
            print("OK")
            print(f"{(sq.tolist())}\n" \
                   f"{(new_labels)}\n" \
                   f"sq: {len(sq)}\n" \
                   f"new labels: {len(new_labels)}")
        else:
            diff = len(sq) - len(new_labels)
            for elem in range(diff):
                new_labels.append(2)
        assert len(sq) == len(new_labels), f"Mismatch.\n" \
                                           f"Text: {text}\n" \
                                           f"{(sq.tolist())}\n" \
                                           f"{(new_labels)}\n" \
                                           f"sq: {len(sq)}\n" \
                                           f"new labels: {len(new_labels)}"
        # tensorize the new labels
        # print(list(zip(sq_as_list, new_labels, [tokenizer.convert_ids_to_tokens(t) for t in sq_as_list])))
        label = torch.tensor(new_labels)
        out_toks_and_labels.append({'input_ids': sq,
                                    'attention_mask': attention_mask,
                                    'labels': label})
    return out_toks_and_labels
