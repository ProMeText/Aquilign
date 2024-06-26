# AQUILIGN -- Mutilingual aligner and collator

[![codecov](https://codecov.io/github/ProMeText/Aquilign/graph/badge.svg?token=TY5HCBOOKL)](https://codecov.io/github/ProMeText/Aquilign)


This repo contains a set of scripts to align and collate a multilingual medieval corpus. Its designers are Matthias Gille Levenson, Lucence Ing and Jean-Baptiste Camps.  

It is based on a fork of the automatic multilingual sentence aligner Bertalign.

The scripts relies for now on a prior phase of text segmentation at syntagm level using regular expressions to match grammatical syntagms and produce a more precise alignment.

## Use

`python3 main.py -o result_dir/ -i data/extraitsLancelot/ii-48/ -mw data/extraitsLancelot/ii-48/castillan/lanzarote-ii-48.txt -d cuda:0`

`python3 main.py --help` to print help.
## Citation

Lei Liu & Min Zhu. 2022. Bertalign: Improved word embedding-based sentence alignment for Chinese–English parallel corpora of literary texts, *Digital Scholarship in the Humanities*. [https://doi.org/10.1093/llc/fqac089](https://doi.org/10.1093/llc/fqac089).


## Licence

This fork is released under the [GNU General Public License v3.0](./LICENCE)

