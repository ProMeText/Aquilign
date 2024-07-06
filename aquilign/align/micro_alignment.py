import transformers
import sys
import itertools
import torch
import json
import re 
import networkx as networkx
import random

model = transformers.BertModel.from_pretrained('google-bert/bert-base-multilingual-cased')
tokenizer = transformers.BertTokenizer.from_pretrained('google-bert/bert-base-multilingual-cased')


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
        

def tokenize_words(sentence:str) -> list:
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
    
    print(len(ids_src))
    print(len(ids_tgt))

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
        
        # Notre matrice de similarité est ici
        dot_prod = torch.matmul(out_src, out_tgt.transpose(-1, -2))
        softmax_srctgt = torch.nn.Softmax(dim=-1)(dot_prod)
        softmax_tgtsrc = torch.nn.Softmax(dim=-2)(dot_prod)

        softmax_inter = (softmax_srctgt > threshold) * (softmax_tgtsrc > threshold)
        print(softmax_inter.shape)
        print(softmax_inter)
        # Sortie: une matrice de tuples de shape (longueur_source, longueur_cible) où chaque valeur 
        # est un booléen correpondant à la similarité des mots.
        
        
    #print("Softmax inter")
    #print(softmax_inter)
    # On récupère la position des valeur True, qui correspondent aux alignements de mots.
    align_subwords = torch.nonzero(softmax_inter, as_tuple=False)
    
    #print("Align subwords:")
    print(align_subwords)
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
        # Ici on vérifie qu'il ne va pas chercher trop loin dans le texte
        if abs(i-j) > 7:
            continue
        alignment_list.append((i, tokenized_src[i], j, tokenized_tgt[j]))
        write_to_log(
            f'[{i}] {tokenized_src[i]}==={tokenized_tgt[j]} [{j}]')
    return alignment_list, tokenized_text, dot_prod


def graph_merge(pair_list):

    G = networkx.petersen_graph()
    G.add_edges_from(pair_list)
    connected_nodes = []
    # On prend chaque noeud et on en ressort les noeuds connectés
    for node in G:
        # https://stackoverflow.com/a/33089602
        connected_components = list(networkx.node_connected_component(G, node))
        connected_components.sort()
        connected_nodes.append(tuple(connected_components))

    # On supprime les noeuds redondants
    connected_nodes = list(set(connected_nodes))
    write_to_log(connected_nodes)
    # Liste de la forme: [('27_a', '27_c', '28_a', '28_c', '29_c', '30_c', '33_b', '34_b'), ('19_c', '20_a', '20_c', '22_b'), etc.]
    return connected_nodes

def main(in_file, threshold):
    with open(".logs/log.txt", "w") as input_log:
        try:
            input_log.truncate(0)
        except FileNotFoundError:
            pass

    with open(in_file, "r") as input_file:
        macro_aligned_text = [line.replace("\n", "").replace("|", "") for line in input_file.readlines()]
        macro_aligned_text = [line.split("\t")[1:] for line in macro_aligned_text]
    device = "cuda:0"
    # random.shuffle(macro_aligned_text)
    n = 70
    n = 149
    for alignment_unit in macro_aligned_text[n:n+1]:
        write_to_log("\n\nNew unit")
        write_to_log(f"Pivot text: {alignment_unit[0]}")
        write_to_log(f"Other texts:")
        write_to_log('\n'.join(alignment_unit[1:]))
        combinaisons = [(alignment_unit[0], alignment_unit[n]) for n in range(1, len(alignment_unit))]
        # combinaisons = [(alignment_unit[0], alignment_unit[n]) for n in range(5, 6)]
        text_dictionnary = {index: tokenize_words(text) for index, text in enumerate(alignment_unit)}
        # combinaisons = list(set(itertools.combinations(alignment_unit, 2)))

        as_index_dict = {}
        as_index_list = []
        all_sents = []
        as_index_list_2 = []
        for index, (search, target) in enumerate(combinaisons):
            write_to_log("---\nNew pair")
            write_to_log(f"Source sentence: {search}")
            write_to_log(f"Target sentence: {target}")
            alignment_results, tokenized_sents, similarities = run_align(search, target, 8, threshold, device)
            alignment_results_as_index = [(f"{unit[0]}-0", f"{unit[2]}-{index + 1}") for unit in alignment_results]
            alignment_results_as_index_2 = [(unit[0], unit[2]) for unit in alignment_results]
            as_index_dict[f"{index + 1}"] = alignment_results_as_index
            as_index_list.extend(alignment_results_as_index)
            all_sents.append(tokenized_sents[1])
            as_index_list_2.append(alignment_results_as_index_2)
            print(similarities)
            print(similarities.shape)
            exit(0)
        print(as_index_dict)
        print(as_index_list_2)
        out_dict = {}
        for index, item in enumerate(as_index_list_2):
            print(item)
            for alignment in item:
                source, target = alignment
                try:
                    out_dict[text_dictionnary[0][source]].append(text_dictionnary[index + 1][target])
                except:
                    out_dict[text_dictionnary[0][source]] = [text_dictionnary[index + 1][target]]
                    
        print(out_dict)
        continue
        nodes = graph_merge(as_index_list)
        print("Nodes created. Creating variants")
        for node in nodes:
            write_to_log("---\nNew variant location:")
            sents = []
            for sentence in node:
                try:
                    word_id, wit = sentence.split("-")
                    sents.append(text_dictionnary[int(wit)][int(word_id)])
                except AttributeError:
                    continue
            write_to_log("\n".join(sents))


if __name__ == '__main__':
    main(sys.argv[1], float(sys.argv[2]))