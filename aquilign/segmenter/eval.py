import json

import evaluate
import numpy as np
import torch

def compute_metrics(predictions, labels, padding_idx):
    """
    This function evaluates the model against the targets.
    :TODO: ignore padding classes?
    :param predictions:
    :param labels:
    :return:
    """
    print("Starting eval")
    predictions = predictions.cpu()
    labels = labels.cpu()
    # load the metrics we want to evaluate
    metric1 = evaluate.load("accuracy")
    metric2 = evaluate.load("recall")
    metric3 = evaluate.load("precision")
    metric4 = evaluate.load("f1")

    # the predictions are of shape [num_example, max_length, out_classes]
    # We reduce the dimensionality of the vector by selecting the higher prob class, on dimension 2
    # This way the out shape is [num_example, max_length]
    predictions = np.argmax(predictions, axis=2)


    # We flatten the 2 vectors to get a 1d vector of shape [num_examples*max_length]
    predictions = np.array(predictions, dtype='int32').flatten()
    labels = np.array(labels, dtype='int32').flatten()

    # On supprime le padding des données
    labels_as_list = labels.tolist()
    predictions_as_list = predictions.tolist()
    predictions = np.array([item for idx, item in enumerate(predictions_as_list) if labels_as_list[idx] != padding_idx], dtype='int32')
    labels = np.array([item for item in labels_as_list if item != padding_idx], dtype='int32')

    # assert 2 not in predictions.tolist(), "Labels reduction didn't work for preds"
    # assert 2 not in labels.tolist(), "Labels reduction didn't work for labels"
    # automatically, value of -100 are produce ; we haven't understood why but we change them to 0. If not, it will give poor results
    ###
    # labels = [0 if x == -100 else x for x in labels]
    ###

    acc = metric1.compute(predictions=predictions, references=labels)
    recall = metric2.compute(predictions=predictions, references=labels, average=None)
    recall_l = []
    [recall_l.extend(v) for k, v in recall.items()]
    precision = metric3.compute(predictions=predictions, references=labels, average=None)
    precision_l = []
    [precision_l.extend(v) for k, v in precision.items()]
    f1 = metric4.compute(predictions=predictions, references=labels, average=None)
    f1_l = []
    [f1_l.extend(v) for k, v in f1.items()]

    print("Eval finished")
    print({"accurracy": acc, "recall": recall_l, "precision": precision_l, "f1": f1_l})
    return {"accurracy": acc, "recall": recall_l, "precision": precision_l, "f1": f1_l}