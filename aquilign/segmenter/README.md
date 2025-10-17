# Segmenter

This repo is under construction. The segmenter will be put aside from the aligner in the near future.

## Installation


Requirements for using the segmenter are in `aquilign/segmenter/requirements.txt`

## Use

### Datasets

The dataset should follow the rules explained in https://anonymous.4open.science/r/Aligning-Medieval-Romance-Text-dataset-D7F8.

### Training

#### Config files

Training hyperparameters should be referenced in the config file in JSON format: 
```JSON
{
  "global": {
    "train": "lsdataset/data/segmented/split/multilingual/train.json",
    "test": "dataset/data/segmented/split/multilingual/test.json",
    "dev": "dataset/data/segmented/split/multilingual/dev.json",
    "import": "/path/to/repo",
    "out_dir": "path/to/results",
    "device": "cuda:0",
    "base_model_name": "google-bert/bert-base-multilingual-cased",
    "data_augmentation": true,
    "freeze_embeddings": false,
    "emb_dim": 768,
    "freeze_lang_embeddings": false,
    "linear_layers": 2,
    "linear_layers_hidden_size": 256,
    "balance_class_weights": true,
    "custom_class_weights": [1.0, 1.2, 0.0],
    "include_lang_metadata": true,
    "lang_emb_dim": 8,
    "epochs": 10,
    "use_pretrained_embeddings": false,
    "use_bert_tokenizer": false,
    "lr": 0.001194,
    "workers": 16,
    "batch_size": 16,
    "segments_max_length": 25,
    "eval_batch_size": 48,
    "use_char_embeddings": true
  },
  "architectures": {
    "lstm": {
    "char_dropout_prob": 0.05,
    "char_embedding_dim": 96,
      "bidirectional": true,
      "lstm_dropout": 0.127,
      "positional_embeddings": false,
      "linear_dropout": 0.30,
      "keep_bert_dimensions": false,
      "lstm_hidden_size": 128,
      "num_lstm_layers": 4,
      "add_attention_layer": true
    }
  }
}

```


#### Commands

The architecture (flag `-a`) to be tested should belong to the following:
- BERT
- DistilBERT
- lstm

The path to the config file is indicated with `-p`. 

A directory will be created using the `-n` flag.

```bash
python3 aquilign/segmenter/trainer.py \
    -a BERT \
    - md google-bert/bert-base-multilingual-cased
    -p aquilign/segmenter/params/BERT.json \
    -n BERT_model
```

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