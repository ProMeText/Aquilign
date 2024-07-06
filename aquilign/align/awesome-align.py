import transformers
import sys
import itertools
import torch
import json
import re


def write_to_log(string):
    print(string)
    with open(".logs/log.txt", "a") as log_file:
        if type(string) != str:
            if type(string) in [list, dict]:
                json.dump(string, log_file)
            else:
                log_file.write(str(string))
        else:
            log_file.write(string)
        log_file.write("\n")


def tokenize_words(sentence: str) -> list:
    """
    Cette fonction tokénise une phrase selon un certain nombre de marqueurs
    """
    words_delimiters = re.compile(r"[\.,;——:\?!’'«»“/\-]|[^\.,;——:\?!’'«»“/\-\s]+")
    sentenceAsList = re.findall(words_delimiters, sentence)
    return sentenceAsList


def remove_punctuation(text: str):
    punct = re.compile(r"[\.,;—:\?!’'«»“/\-]")
    cleaned_text = re.sub(punct, "", text)
    return cleaned_text


def run_align(src, tgt, align_layer, threshold, device):
    print(src)
    print(tgt)
    # pre-processing
    tokenized_src, tokenized_tgt = tokenize_words(src), tokenize_words(tgt)

    token_src, token_tgt = [tokenizer.tokenize(word) for word in tokenized_src], [tokenizer.tokenize(word) for word in
                                                                                  tokenized_tgt]
    wid_src, wid_tgt = [tokenizer.convert_tokens_to_ids(x) for x in token_src], [tokenizer.convert_tokens_to_ids(x) for
                                                                                 x in token_tgt]
    ids_src, ids_tgt = tokenizer.prepare_for_model(list(itertools.chain(*wid_src)), return_tensors='pt',
                                                   model_max_length=tokenizer.model_max_length, truncation=True)[
                           'input_ids'], \
                       tokenizer.prepare_for_model(list(itertools.chain(*wid_tgt)), return_tensors='pt',
                                                   truncation=True, model_max_length=tokenizer.model_max_length)[
                           'input_ids']

    sub2word_map_src = []
    for i, word_list in enumerate(token_src):
        sub2word_map_src += [i for x in word_list]
    sub2word_map_tgt = []
    for i, word_list in enumerate(token_tgt):
        sub2word_map_tgt += [i for x in word_list]

    # alignment
    model.to(device)
    model.eval()
    if device != "cpu":
        ids_src = ids_src.cuda()
        ids_tgt = ids_tgt.cuda()
    with torch.no_grad():
        out_src = model(ids_src.unsqueeze(0), output_hidden_states=True)[2][align_layer][0, 1:-1]
        out_tgt = model(ids_tgt.unsqueeze(0), output_hidden_states=True)[2][align_layer][0, 1:-1]

        dot_prod = torch.matmul(out_src, out_tgt.transpose(-1, -2))

        softmax_srctgt = torch.nn.Softmax(dim=-1)(dot_prod)
        softmax_tgtsrc = torch.nn.Softmax(dim=-2)(dot_prod)

        softmax_inter = (softmax_srctgt > threshold) * (softmax_tgtsrc > threshold)

    align_subwords = torch.nonzero(softmax_inter, as_tuple=False)
    align_subwords.to(device)
    align_words = set()
    for i, j in align_subwords:
        align_words.add((sub2word_map_src[i], sub2word_map_tgt[j]))

    # write_to_loging
    class color:
        PURPLE = '\033[95m'
        CYAN = '\033[96m'
        DARKCYAN = '\033[36m'
        BLUE = '\033[94m'
        GREEN = '\033[92m'
        YELLOW = '\033[93m'
        RED = '\033[91m'
        BOLD = '\033[1m'
        UNDERLINE = '\033[4m'
        END = '\033[0m'

    alignment_list = []
    tokenized_text = (tokenized_src, tokenized_tgt)
    for i, j in sorted(align_words):
        alignment_list.append((i, tokenized_src[i], j, tokenized_tgt[j]))
        write_to_log(
            f'[{i}] {tokenized_src[i]}==={tokenized_tgt[j]} [{j}]')
    return alignment_list, tokenized_text



if __name__ == '__main__':
    model = transformers.BertModel.from_pretrained('google-bert/bert-base-multilingual-cased')
    tokenizer = transformers.BertTokenizer.from_pretrained('google-bert/bert-base-multilingual-cased')
    text_1 = "e vio vn castillo muy bien asentado çercado de muy buen muro todo enderredor e el cato gran pieça el castillo "
    text_2 = "et uoit ung beau chastel monlt bien seant et bien cloz de muraille tout entour. Il regarde le chastel grant pieche "
    run_align(text_1, text_2, threshold=0.08, align_layer=8, device="cuda:0")