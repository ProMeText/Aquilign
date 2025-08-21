import copy
import json
import random

import evaluate
import numpy as np
import torch
import aquilign.segmenter.utils as utils





def compute_ambiguity_metrics(tokens,
                              predictions,
                              labels,
                              id_to_word,
                              word_to_id,
                              output_dir):
    """
    This function produces a confusion matrix for the ambiguous tokens.
    """

    predictions = predictions.cpu()
    labels = labels.cpu()
    tokens = tokens.cpu()
    predictions = np.argmax(predictions, axis=2)
    predictions = np.array(predictions, dtype='int32').flatten()
    tokens = np.array(tokens, dtype='int32').flatten()
    labels = np.array(labels, dtype='int32').flatten()
    ambiguous_tokens = utils.identify_ambiguous_tokens(tokens.tolist(), labels.tolist(), id_to_word, word_to_id)
    recall = evaluate.load("recall")
    precision = evaluate.load("precision")
    f1 = evaluate.load("f1")
    results_per_token = []
    for target_token in ambiguous_tokens:
        target_labels = np.array([label for token, label in zip(tokens, labels) if token == target_token])
        target_predictions = np.array([pred for token, pred in zip(tokens, predictions) if token == target_token])

        current_recall_sc = recall.compute(predictions=target_predictions, references=target_labels, average=None, zero_division=False)["recall"].tolist()[0]
        current_precision_sc = precision.compute(predictions=target_predictions, references=target_labels, average=None, zero_division=False)["precision"].tolist()[0]
        current_f1_sc = f1.compute(predictions=target_predictions, references=target_labels, average=None)["f1"].tolist()[0]
        current_recall_sb = recall.compute(predictions=target_predictions, references=target_labels, average=None, zero_division=False)["recall"].tolist()[1]
        current_precision_sb = precision.compute(predictions=target_predictions, references=target_labels, average=None, zero_division=False)["precision"].tolist()[1]
        current_f1_sb = f1.compute(predictions=target_predictions, references=target_labels, average=None)["f1"].tolist()[1]
        results_per_token.append((id_to_word[target_token], {"precision": [current_recall_sc, current_recall_sb],
                                                             "recall": [current_precision_sc, current_precision_sb],
                                                             "f1": [current_f1_sc, current_f1_sb]}))

    with open(f"{output_dir}/resultats_ambiguite.txt", "w") as output_ambiguity:
        for results in results_per_token:
            recall = ["Recall", results[1]["recall"][0], results[1]["recall"][1]]
            precision = ["Precision", results[1]["precision"][0], results[1]["precision"][1]]
            f1 = ["F1", results[1]["f1"][0], results[1]["f1"][1]]
            header = ["", "Segment Content", "Segment Boundary"]
            output_ambiguity.write(f"Results for {results[0]}:\n\n"
                  f"{utils.format_results(results=[precision, recall, f1], header=header, print_to_term=False)}"
                  f"\n\n\n")




def compute_metrics(predictions,
                    labels,
                    examples,
                    id_to_word,
                    idx_to_class,
                    padding_idx,
                    batch_size,
                    last_epoch=False,
                    tokenizer=None):
    """
    This function evaluates the model against the targets.
    :TODO: ignore padding classes?
    :param predictions:
    :param labels:
    :return:
    """
    # the predictions are of shape [num_example, max_length, out_classes]
    # We reduce the dimensionality of the vector by selecting the higher prob class, on dimension 2
    # This way the out shape is [num_example, max_length]
    predictions = predictions.cpu()
    labels = labels.cpu()
    predictions_as_probs = copy.deepcopy(predictions)
    predictions = np.argmax(predictions, axis=2)
    # On teste un exemple pour voir si tout est OK.
    # TODO: il faut le faire sur le meilleur modèle, pas le dernier
    if last_epoch:
        if tokenizer:
            id_to_word = {ident: value for value, ident in tokenizer.get_vocab().items()}
        examples_number = 10
        random_number = random.randint(0, len(examples) - examples_number)
        example_range = range(random_number, random_number + examples_number)
        print(f"Showing example {random_number} to {random_number + examples_number}:")
        for idx in example_range:
            example = examples[idx].tolist()[1:]
            example_as_string = " ".join([id_to_word[ident] for ident in example]).replace(" ##", "")
            print(example_as_string)
            label = labels[idx].tolist()[1:]
            position_first_left_padding = next(index for index, ident in enumerate(example) if ident == 0)
            example_no_padding = example[:position_first_left_padding]
            label_no_padding = label[:position_first_left_padding]

            probs_no_padding = predictions_as_probs[idx].tolist()[1:position_first_left_padding + 1]
            corresp_prediction = predictions[idx].tolist()[1:position_first_left_padding + 1]
            corresp_prediction_as_classes = [item for item in corresp_prediction]
            corresp_label_as_classes = [item for item in label_no_padding]
            corresp_tokens_as_str = [id_to_word[item] for item in example_no_padding]
            correct = []
            for pred, label in zip(corresp_prediction_as_classes, corresp_label_as_classes):
                if label == 1:
                    if pred == label:
                        correct.append(True)
                    else:
                        correct.append(False)
                else:
                    if pred == 1:
                        correct.append(False)
                    else:
                        correct.append("")

            assert len(corresp_prediction) == len(example_no_padding) == len(corresp_tokens_as_str) == len(correct)
            for ex, token, prediction, target, correct, prob in list(
                    zip(example_no_padding,
                        corresp_tokens_as_str,
                        corresp_prediction_as_classes,
                        corresp_label_as_classes,
                        correct,
                        probs_no_padding)
            ):
                print(f"{ex}\t{token}\t{prediction}\t{target}\t{correct}\t{prob}")
            print("---")


    # load the metrics we want to evaluate
    metric1 = evaluate.load("accuracy")
    metric2 = evaluate.load("recall")
    metric3 = evaluate.load("precision")
    metric4 = evaluate.load("f1")


    # We flatten the 2 vectors to get a 1d vector of shape [num_examples*max_length]
    predictions = np.array(predictions, dtype='int32').flatten()
    labels = np.array(labels, dtype='int32').flatten()

    # On supprime le padding des données
    labels_as_list = labels.tolist()
    predictions_as_list = predictions.tolist()
    # predictions = np.array([item for idx, item in enumerate(predictions_as_list) if labels_as_list[idx] != padding_idx], dtype='int32')
    # labels = np.array([item for item in labels_as_list if item  != padding_idx], dtype='int32')

    # assert 2 not in predictions.tolist(), "Labels reduction didn't work for preds"
    # assert 2 not in labels.tolist(), "Labels reduction didn't work for labels"
    # automatically, value of -100 are produce ; we haven't understood why but we change them to 0. If not, it will give poor results
    ###
    # labels = [0 if x == -100 else x for x in labels]
    ###
    print(len(predictions_as_list))
    print(len(labels_as_list))

    predictions = np.array([item for idx, item in enumerate(predictions_as_list) if labels_as_list[idx]  != 2], dtype='int32')
    labels = np.array([item for item in labels_as_list if item != 2], dtype='int32')

    accuracy = metric1.compute(predictions=predictions, references=labels)
    recall = metric2.compute(predictions=predictions, references=labels, average=None)
    recall_l = []
    [recall_l.extend(v) for k, v in recall.items()]
    precision = metric3.compute(predictions=predictions, references=labels, average=None)
    precision_l = []
    [precision_l.extend(v) for k, v in precision.items()]
    f1 = metric4.compute(predictions=predictions, references=labels, average=None)
    f1_l = []
    [f1_l.extend(v) for k, v in f1.items()]

    results = {"accuracy": accuracy, "recall": recall_l, "precision": precision_l, "f1": f1_l}
    return results