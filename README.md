# 📐 AQUILIGN – Multilingual Aligner and Collator

[![codecov](https://codecov.io/github/ProMeText/Aquilign/graph/badge.svg?token=TY5HCBOOKL)](https://codecov.io/github/ProMeText/Aquilign)
[![Last Commit](https://img.shields.io/github/last-commit/ProMeText/Aquilign)](https://github.com/ProMeText/Aquilign/commits/main)
[![Repo Size](https://img.shields.io/github/repo-size/ProMeText/Aquilign)](https://github.com/ProMeText/Aquilign)
[![Issues](https://img.shields.io/github/issues/ProMeText/Aquilign)](https://github.com/ProMeText/Aquilign/issues)
[![Forks](https://img.shields.io/github/forks/ProMeText/Aquilign)](https://github.com/ProMeText/Aquilign/network/members)
[![Stars](https://img.shields.io/github/stars/ProMeText/Aquilign)](https://github.com/ProMeText/Aquilign/stargazers)

**AQUILIGN** is a multilingual alignment and collation engine designed for **historical and philological corpora**.  
It performs **clause-level alignment** of parallel texts using a combination of **regular-expression and BERT-based segmentation**, and supports multilingual workflows across medieval Romance, Latin, and Middle English texts.

🧪 Developed by [Matthias Gille Levenson](https://github.com/matgille), [Lucence Ing](https://cv.hal.science/lucence-ing), and [Jean-Baptiste Camps](https://github.com/Jean-Baptiste-Camps).  
Originally presented at the *Computational Humanities Research Conference (CHR 2023)* — see [citation](#citation) for full reference.


---

## 💡 Key Features

- 🔀 **Multilingual clause-level alignment** using contextual embeddings  
- ✂️ **Trainable segmentation module** (BERT-based or regex-based)  
- 🧩 **Collation-ready architecture** (stemmatology support in development)  
- 📚 Optimized for **premodern and historical corpora**

AQUILIGN builds on a fork of [Bertalign](https://github.com/bfsujason/bertalign), customized for historical languages and alignment evaluation.

---

## ⚙️ Installation

> ⚠️ **Caveat**: AQUILIGN is currently tested on **Python 3.9 and 3.10** due to certain library constraints.  
> Compatibility with other versions is not guaranteed.

```bash
git clone https://github.com/ProMeText/Aquilign.git
cd Aquilign
pip install -r requirements.txt
```
## 🧠 Training the Segmenter

The segmenter is based on a trainable `BertForTokenClassification` model from Hugging Face’s `transformers` library.

We fine-tune this model to detect custom sentence delimiters (`£`) in historical texts from the **[Multilingual Segmentation Dataset](https://github.com/carolisteia/multilingual-segmentation-dataset)**.

---

### 🔧 Example Command

```bash
python3 train_tokenizer.py \
  -m google-bert/bert-base-multilingual-cased \
  -t multilingual-segmentation-dataset/data/Multilingual_Aegidius/segmented/split/multilingual/train.json \
  -d multilingual-segmentation-dataset/data/Multilingual_Aegidius/segmented/split/multilingual/dev.json \
  -e multilingual-segmentation-dataset/data/Multilingual_Aegidius/segmented/split/multilingual/test.json \
  -ep 100 \
  -b 128 \
  --device cuda:0 \
  -bf16 \
  -n multilingual_model \
  -s 2 \
  -es 10
```
This command fine-tunes the `bert-base-multilingual-cased` model with the following configuration:

- **Epochs**: `100`  
- **Batch size**: `128`  
- **Device**: `cuda:0` (GPU)  
- **Precision**: `bf16` (bfloat16 mixed precision)  
- **Checkpointing**: Saves the model every 2 epochs  
- **Early stopping**: Stops after 10 epochs without improvement

---

### 🗂️ Input Format: JSON Schema

Training data must follow a structured JSON format, including both metadata and examples.

```json
{
  "metadata": {
    "lang": ["la", "it", "es", "fr", "en", "ca", "pt"],
    "centuries": [13, 14, 15, 16],
    "delimiter": "£"
  },
  "examples": [
    {
      "example": "que mi padre me diese £por muger a un su fijo del Rey",
      "lang": "es"
    },
    {
      "example": "Per fé, disse Lion, £i v’andasse volentieri, £ma i vo veggio £qui",
      "lang": "it"
    }
  ]
}
```
- The `metadata` block must include:

  - `"lang"`: a list of ISO 639-1 codes representing the languages in the dataset  
  - `"centuries"`: historical coverage of the examples (used for metadata and possible filtering)  
  - `"delimiter"`: the segmentation marker token (default: `£`), predicted by the model

- The `examples` block is an array of training samples, each containing:

  - `"example"`: a string of text including segmentation markers  
  - `"lang"`: the ISO code of the language the text belongs to

---

📖 For more details, see the full documentation:  
➡️ [segmentation_model.md](https://github.com/carolisteia/multilingual-segmentation-dataset/blob/main/docs/segmentation_model.md)


## 🧮 Using the Aligner

To align a set of parallel texts using the BERT-based segmenter, run:

```bash
python3 main.py \
  -o lancelot \
  -i data/extraitsLancelot/ii-48/ \
  -mw data/extraitsLancelot/ii-48/fr/micha-ii-48.txt \
  -d cuda:0 \
  -t bert-based
```
This will:

- ✅ Align the multilingual files found in `data/extraitsLancelot/ii-48/`
- 📚 Use the **Micha edition** (French) as the **base witness**
- ⚙️ Run on the **GPU** (`cuda:0`)
- 💾 Save results to: `result_dir/lancelot/`


> 📂 Files must be sorted by language, using the ISO 639-1 language code  
> as the **parent directory name** (`es/`, `fr/`, `it/`, `en/`, etc.).

To view all available options:

```bash
python3 main.py --help
```

```

## 🔗 Related Projects

**Aquilign** is part of a broader ecosystem of tools and corpora developed for the computational study of medieval multilingual textual traditions. The following repositories provide aligned datasets, segmentation resources, and use cases for the Aquilign pipeline:

- [Multilingual Segmentation Dataset](https://github.com/carolisteia/multilingual-segmentation-dataset)
  Sentence and clause-level segmentation datasets in seven medieval languages, used to train and evaluate the segmentation model integrated into Aquilign.

- [Parallelium – an aligned scriptures dataset](https://github.com/carolisteia/parallelium-scriptures-alignment-dataset)  
  A multilingual dataset of aligned Biblical and Qur’anic texts (medieval and modern), used for benchmarking multilingual alignment in diverse historical settings.

- [Lancelot par maints langages](https://github.com/carolisteia/lancelot-par-maints-langages)  
  A parallel corpus of *Lancelot en prose* in French, Castilian, and Italian. First testbed for Aquilign’s multilingual alignment and stemmatological comparison.

- [Multilingual Aegidius](https://github.com/ProMeText/Multilingual_Aegidius)  
  A corpus of *De regimine principum* and its translations in Latin, Romance vernaculars, and Middle English. Built using the Aquilign segmentation and alignment workflow.

---

## 🚧 Project Status & Future Directions

**Aquilign** is under active development and currently supports:

- ✅ Sentence- and clause-level alignment across multiple languages  
- ✅ Integration with BERT-based and regex-based segmenters  
- ✅ Alignment evaluation and output export in tabular format  
- ✅ Compatibility with multilingual historical corpora (e.g. *Lancelot*, *De Regimine Principum*)

---

### 🔮 Planned Features

- 🧬 **Collation Module**:  
  Automatic generation of collation tables across aligned witnesses for textual variant analysis

- 🏛️ **Stemmatic Analysis Integration**:  
  Tools for stemmatological inference based on alignment structure and textual divergence

- 📊 **Interactive Visualization Tools**:  
  Visualization of alignment, variant graphs, and stemma hypotheses

- 🌐 **Support for Additional Languages**:  
  Extending tokenization and alignment capabilities to new premodern languages and scripts

---

If you're interested in contributing to any of these areas or proposing enhancements, see [Contact & Contributions](#-contact--contributions).

---

## 📫 Contact & Contributions

We welcome questions, feedback, and contributions to improve the Aquilign pipeline.

- 🛠️ Found a bug or have a feature request?  
  ➡️ [Open an issue](https://github.com/ProMeText/Aquilign/issues)

- 🔄 Want to contribute code or improvements?  
  ➡️ Fork the repo and submit a pull request

- 🎓 For academic collaboration or project inquiries:  
  ➡️ Reach out via [GitHub Discussions](https://github.com/ProMeText/Aquilign/discussions) or contact the authors directly

---

--
## 📚 Citation

If you use this tool in your research, please cite:

Gille Levenson, M., Ing, L., & Camps, J.-B. (2024).  
**Textual Transmission without Borders: Multiple Multilingual Alignment and Stemmatology of the _Lancelot en prose_ (Medieval French, Castilian, Italian).**  
In W. Haverals, M. Koolen, & L. Thompson (Eds.), *Proceedings of the Computational Humanities Research Conference 2024* (Vol. 3834, pp. 65–92). CEUR.  
🔗 [https://ceur-ws.org/Vol-3834/#paper104](https://ceur-ws.org/Vol-3834/#paper104)

### 📄 BibTeX

```bibtex
@inproceedings{gillelevenson_TextualTransmissionBorders_2024a,
  title = {Textual Transmission without Borders: Multiple Multilingual Alignment and Stemmatology of the ``Lancelot En Prose'' (Medieval French, Castilian, Italian)},
  shorttitle = {Textual Transmission without Borders},
  booktitle = {Proceedings of the Computational Humanities Research Conference 2024},
  author = {Gille Levenson, Matthias and Ing, Lucence and Camps, Jean-Baptiste},
  editor = {Haverals, Wouter and Koolen, Marijn and Thompson, Laure},
  date = {2024},
  series = {CEUR Workshop Proceedings},
  volume = {3834},
  pages = {65--92},
  publisher = {CEUR},
  location = {Aarhus, Denmark},
  issn = {1613-0073},
  url = {https://ceur-ws.org/Vol-3834/#paper104},
  urldate = {2024-12-09},
  eventtitle = {Computational Humanities Research 2024},
  langid = {english}
}
```
--- 
## 💰 Funding

This work benefited from national funding managed by the **Agence Nationale de la Recherche**  
under the *Investissements d'avenir* programme with the reference:  
**ANR-21-ESRE-0005 (Biblissima+)**

> Ce travail a bénéficié d'une aide de l’État gérée par l’**Agence Nationale de la Recherche**  
> au titre du programme d’**Investissements d’avenir**, référence **ANR-21-ESRE-0005 (Biblissima+)**.

<p align="center">
  <img src="https://github.com/user-attachments/assets/915c871f-fbaa-45ea-8334-2bf3dde8252d" alt="Biblissima+ Logo" width="600"/>
</p>

## ⚖️ License

This project is released under the **[GNU General Public License v3.0](./LICENCE)**.  
You are free to use, modify, and redistribute the code under the same license conditions.
