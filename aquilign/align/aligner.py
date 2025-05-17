import numpy as np

import aquilign.align.corelib as core
import aquilign.align.utils as utils
import torch.nn as nn
import torch

class Bertalign_Embbed:
    """
    Cette classe implémente un outil de plongement simple de phrases
    """
    def __init__(self,
                 model,
                 sents,
                 max_align):
        print("Embbedding source text.")
        self.sents_vecs, self.src_lens = model.transform(sents, max_align - 1)
        self.search_simple_vecs = model.simple_vectorization(self.sents_vecs)

    def return_embbeds(self):
        return self.sents_vecs, self.src_lens, self.search_simple_vecs

class Bertalign:
    def __init__(self,
                 model,
                 src_sents=None,
                 src_vecs=None,
                 src_lens=None,
                 search_simple_vecs=None,
                 src=None,
                 tgt=None,
                 max_align=3,
                 top_k=3,
                 win=5,
                 skip=-0.1,
                 margin=True,
                 len_penalty=True,
                 is_split=False,
                 device="cpu"):



        if tgt is None:
            tgt = []
        if src is None:
            src = []
        self.max_align = max_align
        self.top_k = top_k
        self.win = win
        self.skip = skip
        self.margin = margin
        self.len_penalty = len_penalty
        self.device = device
        self.model = model

        tgt_sents = tgt

        src_num = len(src_sents)
        tgt_num = len(tgt_sents)
        assert len(src_sents) != 0, "Problemo"

        if not src_sents and not src_lens and not src_vecs and not search_simple_vecs:
            print("Embedding target and source text using {} ...".format(model.model_name))
            src_vecs, src_lens = self.model.transform(src_sents, max_align - 1)
            self.src_simple_vecs = self.model.simple_vectorization(src_sents)

        else:
            print("Embedding target text using {} ...".format(model.model_name))
            tgt_vecs, tgt_lens = self.model.transform(tgt_sents, max_align - 1)
            self.search_simple_vecs = search_simple_vecs

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
        D, I = core.find_top_k_sents(self.src_vecs[0, :], self.tgt_vecs[0, :], k=self.top_k, device="cpu")
        first_alignment_types = core.get_alignment_types(2)  # 0-1, 1-0, 1-1
        first_w, first_path = core.find_first_search_path(self.src_num, self.tgt_num)
        first_pointers = core.first_pass_align(self.src_num, self.tgt_num, first_w, first_path, first_alignment_types,
                                               D, I)
        first_alignment = core.first_back_track(self.src_num, self.tgt_num, first_pointers, first_path,
                                                first_alignment_types)

        print("Performing second-step alignment ...")
        second_alignment_types = core.get_alignment_types(self.max_align)
        second_w, second_path = core.find_second_search_path(first_alignment, self.win, self.src_num, self.tgt_num)
        second_pointers = core.second_pass_align(self.src_vecs, self.tgt_vecs, self.src_lens, self.tgt_lens,
                                                 second_w, second_path, second_alignment_types,
                                                 self.char_ratio, self.skip, margin=self.margin,
                                                 len_penalty=self.len_penalty)
        second_alignment = core.second_back_track(self.src_num, self.tgt_num, second_pointers, second_path,
                                                  second_alignment_types)

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
            line = ' '.join(lines[bead[0]:bead[-1] + 1])
        return line
