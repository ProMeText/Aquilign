import numpy as np

import aquilign.align.corelib as core
import aquilign.align.utils as utils
import aquilign.align.graph_merge as graph_merge
import torch.nn as nn
import torch

class Bertalign:
    def __init__(self,
                 model,
                 src,
                 tgt,
                 max_align=3,
                 top_k=3,
                 win=5,
                 skip=-0.1,
                 margin=True,
                 len_penalty=True,
                 is_split=False,
                 device="cpu"):
        
        self.max_align = max_align
        self.top_k = top_k
        self.win = win
        self.skip = skip
        self.margin = margin
        self.len_penalty = len_penalty
        self.device = device
        self.model = model
        
        
    
        
        src_sents = src
        tgt_sents = tgt
        # print(src_sents)
        # print(tgt_sents)
 
        src_num = len(src_sents)
        tgt_num = len(tgt_sents)
        assert len(src_sents) != 0, "Problemo"

        print("Embedding source and target text using {} ...".format(model.model_name))
        src_vecs, src_lens = self.model.transform(src_sents, max_align - 1)
        tgt_vecs, tgt_lens = self.model.transform(tgt_sents, max_align - 1)
        
        self.search_simple_vecs = self.model.simple_vectorization(src_sents)
        self.tgt_simple_vecs = self.model.simple_vectorization(tgt_sents)

        char_ratio = np.sum(src_lens[0,]) / np.sum(tgt_lens[0,])

        self.src_sents = src_sents
        self.tgt_sents = tgt_sents
        self.src_num = src_num
        self.tgt_num = tgt_num
        self.src_lens = src_lens
        self.tgt_lens = tgt_lens
        self.char_ratio = char_ratio
        self.src_vecs = src_vecs
        self.tgt_vecs = tgt_vecs
    
    def compute_distance(self):
        if torch.cuda.is_available() and self.device == 'cuda:0':  # GPU version
            cos = nn.CosineSimilarity(dim=1, eps=1e-6)
            output = cos(torch.from_numpy(self.search_simple_vecs), torch.from_numpy(self.tgt_simple_vecs))
        else:
            print("Code to run on CPU not implemented. Exiting")
            exit(0)
        return output

    
    
    def align_sents(self, first_alignment_only=False):
        print("Performing first-step alignment ...")
        D, I = core.find_top_k_sents(self.src_vecs[0,:], self.tgt_vecs[0,:], k=self.top_k, device=self.device)
        first_alignment_types = core.get_alignment_types(2) # 0-1, 1-0, 1-1
        first_w, first_path = core.find_first_search_path(self.src_num, self.tgt_num)
        first_pointers = core.first_pass_align(self.src_num, self.tgt_num, first_w, first_path, first_alignment_types, D, I)
        first_alignment = core.first_back_track(self.src_num, self.tgt_num, first_pointers, first_path, first_alignment_types)
        
        print("Performing second-step alignment ...")
        second_alignment_types = core.get_alignment_types(self.max_align)
        second_w, second_path = core.find_second_search_path(first_alignment, self.win, self.src_num, self.tgt_num)
        second_pointers = core.second_pass_align(self.src_vecs, self.tgt_vecs, self.src_lens, self.tgt_lens,
                                            second_w, second_path, second_alignment_types,
                                            self.char_ratio, self.skip, margin=self.margin, len_penalty=self.len_penalty)
        second_alignment = core.second_back_track(self.src_num, self.tgt_num, second_pointers, second_path, second_alignment_types)
        
        if first_alignment_only:
            self.result = first_alignment
        else:
            self.result = second_alignment
    
    def print_sents(self):
        for bead in (self.result):
            src_line = self._get_line(bead[0], self.src_sents)
            tgt_line = self._get_line(bead[1], self.tgt_sents)
            print(bead)
            print(src_line + "\n" + tgt_line + "\n")

    @staticmethod
    def _get_line(bead, lines):
        line = ''
        if len(bead) > 0:
            line = ' '.join(lines[bead[0]:bead[-1]+1])
        return line


def realign_units(alignments, model):
    # first we search for all following units with the main wit omitted
    # print(alignments)
    empty_units = [(idx, item) for idx, item in enumerate(alignments) if item['a'] == []]
    # we then search for following entries
    all_indices = [(idx, unit[0]) for idx, unit in enumerate(empty_units)]
    filtered = [position for idx, (position, true_index) in enumerate(all_indices[:-1]) if all_indices[idx + 1][1] == true_index + 1 or all_indices[idx - 1][1] == true_index - 1]
    empty_units = [empty_units[idx] for idx in filtered]
    # Let's group the alignment units by adjacent positions
    groups_dict = {0: [empty_units[0]]}
    group = 0
    for idx, item in enumerate(empty_units[1:]):
        position, _ = item
        if groups_dict[group][-1][0] == position - 1:
            pass
        else:
            group += 1
        try:
            groups_dict[group].append(item)
        except KeyError:
            groups_dict[group] = [item]
            
    # We then remove the groups that have the exact same wits through the units omitted by the pivot wit (= there is nothing to do)
    filtered_dict = {}
    for key, value in groups_dict.items():
        concerned_wits = []
        wits_as_dict = {}
        for _, item in value:
            concerned_wits.extend([wit for wit in item.keys()])
            for sigla, val in item.items():
                try:
                    if len(val) != 0:
                        wits_as_dict[sigla] += 1
                except KeyError:
                    wits_as_dict[sigla] = 1
        if len(list(set(concerned_wits))) != 1 and len(wits_as_dict) != 1:
            filtered_dict[key] = value
    # we extend the dict
    for key, value in filtered_dict.items():
        print("---")
        first_item = value[0][0]
        previous_item = first_item - 2
        last_item = value[-1][0]
        next_item = first_item + 2
        previous_items = [(i, alignments[i]) for i in range(previous_item, first_item)]
        next_items = [(i, alignments[i]) for i in range(last_item + 1, next_item + 1)]
        filtered_dict[key] = previous_items + value + next_items
        print(previous_item)
        print(next_item)
        print(filtered_dict[key])
        print(key)
    
    path_list = [utils.read_json('result_dir/test_recreate/tokenized_micha-ii-48.json'),
                 utils.read_json('result_dir/test_recreate/tokenized_fr751-ii-48.json'),
                 utils.read_json('result_dir/test_recreate/tokenized_sommer--ii-48.json'),
                 utils.read_json('result_dir/test_recreate/tokenized_fr111-ii-48.json'),
                 utils.read_json('result_dir/test_recreate/tokenized_inc-ii-48.json'),
                 utils.read_json('result_dir/test_recreate/tokenized_lanzarote-ii-48.json'),
                 utils.read_json('result_dir/test_recreate/tokenized_lancellotto-ii-48.json')]
    
    new_corpus = {key: {} for key in filtered_dict.keys()}
    mapping = {letter: index for index, letter in zip(range(10), "abcdefghij")}
    for key, value in filtered_dict.items():
        for idx, item in value:
            for k, v in item.items():
                try:
                    new_corpus[key][k].extend(path_list[mapping[k]][int(it)] for it in v)
                except KeyError:
                    new_corpus[key][k] = [path_list[mapping[k]][int(it)] for it in v]
    all_updated_alignments = []
    print(len(new_corpus))
    
    alignment_dicts = dict()
    for key, value in new_corpus.items():
        print(f"---\nNew unit: {key}")
        all_texts = list(value.values())
        print([(idx, item) for idx, item in enumerate(all_texts)])
        print([(idx, len(text)) for idx, text in enumerate(all_texts)])
        new_pivot = all_texts.pop(1)
        all_sigla = [item for item in "abcdefghij"]
        old_sigla = all_sigla.copy()
        new_pivot_sigla = [all_sigla.pop(1)]
        new_sigla = new_pivot_sigla + all_sigla
        new_mapping = {letter: index for index, letter in zip(new_sigla, old_sigla)}
        align_dict = dict()
        for index, text in enumerate(all_texts):
            new_aligner = Bertalign(model,
                                    new_pivot,
                                    text,
                                    max_align=2,
                                    win=5, 
                                    skip=-.1,
                                    margin=True,
                                    len_penalty=True,
                                    device="cuda:0")
            new_aligner.align_sents()
            align_dict[index] = new_aligner.result
        alignment_dicts[key] = align_dict
        list_of_merged_alignments = graph_merge.merge_alignment_table(align_dict, rerun_alignments=True)
        all_updated_alignments.append((key, list_of_merged_alignments))
    
    # We reverse the list to replace with correct alignments
    all_updated_alignments.sort(key=lambda x: x[0], reverse=True)
    print("Updating list")
    for key, list_of_merged_alignments in all_updated_alignments:
        print(f"--- key: {key}")
        first_pos_dict = {key: value[0] for key, value in filtered_dict[key][0][1].items()}
        updated_list_of_alignments = []
        for alignment in list_of_merged_alignments:
            new_alignment = {new_mapping[wit]:[str(int(val) + int(first_pos_dict[new_mapping[wit]])) for val in value] for wit, value in alignment.items()}
            updated_list_of_alignments.append(new_alignment)
        first_item = filtered_dict[key][0][0]
        last_item = filtered_dict[key][-1][0]
        length = len(filtered_dict[key])
        print(f"As alignement: {alignment_dicts[key]}")
        print(f"Original: {filtered_dict[key]}")
        print(f"Replacement: {updated_list_of_alignments}")
        all_to_remove = [item for item in range(first_item, last_item)]
        [alignments.pop(first_item) for i in all_to_remove]
        alignments = alignments[:first_item] + updated_list_of_alignments + alignments[first_item + 1:]
        utils.write_json("/home/mgl/Documents/updated_dict.json", alignments)
        
    return alignments
                                                                                                         