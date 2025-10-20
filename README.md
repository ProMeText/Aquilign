# 📐 Multilingual Medieval Text Segmenter


**XXX** is a multilingual alignment and collation engine designed for **historical and philological corpora**.  
It performs **clause-level alignment** of parallel texts using a combination of **regular-expression and BERT-based segmentation**, and supports multilingual workflows across medieval Romance, Latin, and Middle English texts.

This repository 

🧪 Developed by .  
Originally presented at the *XXX Conference (CHR 2023)*.


---

## 💡 Key Features

- 🔀 **Multilingual clause-level segmentation** using contextual embeddings

The segmenter is part of a larger project built on a fork of [Bertalign](https://github.com/bfsujason/bertalign), 
customized for historical languages and alignment evaluation.

It is an intermediary version made for the LREC2026 conference. The aligner works but is 
not the main objective of this repository.

---

## ⚙️ Installation

> ⚠️ **Caveat**: The aligner is currently tested on **Python 3.9 and 3.10** due to certain library constraints.  
> Compatibility with other versions is not guaranteed.

```bash
git clone 
cd Aquilign
pip install -r aquilign/segmenter/requirements.txt
```
## 🧠 Training the Segmenter

The segmenter is based on a trainable `BertForTokenClassification` model from Hugging Face’s `transformers` library.

We fine-tune this model to detect custom sentence delimiters in historical texts from the **[Multilingual Segmentation Dataset](https://github.com/carolisteia/multilingual-segmentation-dataset)**.

```bash
python3 aquilign/segmenter/trainer.py \
        -m train \
        -a BERT \
        -p path/to/params/CBP_BERT.json \
        -n training_session_name
```

The config file should have the following shape:
```json
{
  "global": {
     "train": "path/to/train.json",
    "test": "path/to/test.json",
    "dev": "path/to/dev.json",
    "import": "/path/to/repo", 
    "base_model_name": "google-bert/bert-base-multilingual-cased",
    "out_dir": "/path/to/out/dir",  "device": "cuda:0",
    "data_augmentation": true,
    "epochs": 10,
    "lr": 0.00001,
    "workers": 8,
    "batch_size": 128,
    "eval_batch_size": 128
  }
}
```

The training creates several files and result files. 
The model config (epochs, batch sizes, learning rate) is written in the output directory (`./config/config.json`) 
as a logging file.

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
      "example": "£Per fé, disse Lion, £i v’andasse volentieri, £ma i vo veggio £qui",
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


### Tests

The `-m` flag should be used for testing. The model to test has to be indicated with the `-md` flag. It can be:
- a huggingface name
- a link to the BERT local model (link to the directory)
- a link to the local LSTM model  (link to the model). In this case, the used vocabularies should 
be indicated with the `-v` flag. As there is an input vocabulary and a lang vocabulary (for models trained with 
- lang metadata info), the path should be to the vocab dir. 

```bash
python3 aquilign/segmenter/trainer.py \
        -m test \
        -md paper_LREC/models/LSTM_WordEmbs/models/best/best.pt  \
        -v /paper_LREC/models/LSTM_CharEmbs/vocab \
        -a lstm \
        -p  /paper_LREC/models/LSTM_CharEmbs/config/config.json \
        -o /path/to/out/dir
```


### Inference


---
## 🔗 Related Projects

**XXX** is part of a broader ecosystem of tools and corpora developed for the computational study of medieval multilingual textual traditions. The following repositories provide aligned datasets, segmentation resources, and use cases for the Aquilign pipeline:

- [Multilingual Segmentation Dataset](ANONYMOUS)
  Sentence and clause-level segmentation datasets in seven medieval languages, used to train and evaluate the segmentation model integrated into Aquilign.

---

## 🚧 Project Status & Future Directions

---


---

## 📫 Contact & Contributions

We welcome questions, feedback, and contributions to improve the XXX pipeline.

---
## 📚 Citation

If you use this tool in your research, please cite:


### 📄 BibTeX

--- 
## 💰 Funding

## ⚖️ License

This project is released under the **[GNU General Public License v3.0](./LICENCE)**.  
You are free to use, modify, and redistribute the code under the same license conditions.
