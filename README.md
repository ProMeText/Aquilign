# AQUILIGN -- Mutilingual aligner and collator

[![codecov](https://codecov.io/github/ProMeText/Aquilign/graph/badge.svg?token=TY5HCBOOKL)](https://codecov.io/github/ProMeText/Aquilign)


This repo contains a set of scripts to align (and soon collate) a multilingual medieval corpus. Its designers are Matthias Gille Levenson, Lucence Ing and Jean-Baptiste Camps.  

It is based on a fork of the automatic multilingual sentence aligner Bertalign.

The scripts relies on a prior phase of text segmentation at syntagm level using regular expressions or bert-based segmentation to match grammatical syntagms and produce a more precise alignment.

## Use

`python3 main.py -o lancelot -i data/extraitsLancelot/ii-48/ -mw data/extraitsLancelot/ii-48/fr/micha-ii-48.txt -d 
cuda:0 -t bert-based` to perform alignment with our bert-based segmenter, choosing Micha edition as base witness,
on the GPU. The results will be saved in `result_dir/lancelot`

`python3 main.py --help` to print help.

Files must be sorted by language, using the ISO_639-1 language code as parent directory name (`es`, `fr`, `it`, `en`, etc).
## Citation

Gille Levenson, M., Ing, L., & Camps, J.-B. (2024). Textual Transmission without Borders: Multiple Multilingual Alignment and Stemmatology of the ``Lancelot en prose’’ (Medieval French, Castilian, Italian). In W. Haverals, M. Koolen, & L. Thompson (Eds.), Proceedings of the Computational Humanities   Research Conference 2024 (Vol. 3834, pp. 65–92). CEUR. https://ceur-ws.org/Vol-3834/#paper104


```
@inproceedings{gillelevenson_TextualTransmissionBorders_2024a,
  title = {Textual {{Transmission}} without {{Borders}}: {{Multiple Multilingual Alignment}} and {{Stemmatology}} of the ``{{Lancelot}} En Prose'' ({{Medieval French}}, {{Castilian}}, {{Italian}})},
  shorttitle = {Textual {{Transmission}} without {{Borders}}},
  booktitle = {Proceedings of the {{Computational Humanities}}   {{Research Conference}} 2024},
  author = {Gille Levenson, Matthias and Ing, Lucence and Camps, Jean-Baptiste},
  editor = {Haverals, Wouter and Koolen, Marijn and Thompson, Laure},
  date = {2024},
  series = {{{CEUR Workshop Proceedings}}},
  volume = {3834},
  pages = {65--92},
  publisher = {CEUR},
  location = {Aarhus, Denmark},
  issn = {1613-0073},
  url = {https://ceur-ws.org/Vol-3834/#paper104},
  urldate = {2024-12-09},
  eventtitle = {Computational {{Humanities Research}} 2024},
  langid = {english},
  file = {/home/mgl/Bureau/Travail/Bibliotheque_zoteros/storage/CIH7IAHV/Levenson et al. - 2024 - Textual Transmission without Borders Multiple Multilingual Alignment and Stemmatology of the ``Lanc.pdf}
}

```


## Licence

This fork is released under the [GNU General Public License v3.0](./LICENCE)

## Funding

This work benefited́ from national funding managed by the Agence Nationale de la Recherche under the Investissements d'avenir programme with the reference ANR-21-ESRE-0005 (Biblissima+). 

Ce travail a bénéficié́ d'une aide de l’État gérée par l'Agence Nationale de la Recherche au titre du programme d’Investissements d’avenir portant la référence ANR-21-ESRE-0005 (Biblissima+) 

![image](https://github.com/user-attachments/assets/915c871f-fbaa-45ea-8334-2bf3dde8252d)

