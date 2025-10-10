import copy
import json
import random
import statistics

import evaluate
import numpy as np
import torch
import aquilign.segmenter.utils as utils





def compute_ambiguity_metrics(tokens,
                              predictions,
                              labels,
                              id_to_word,
                              word_to_id,
                              log_dir,
                              name=None):
    """
    This function produces a confusion matrix for the ambiguous tokens.
    """
    print("Computing ambiguity metrics")
    predictions = predictions.cpu()
    labels = labels.cpu()
    tokens = tokens.cpu()
    predictions = np.argmax(predictions, axis=2)
    predictions = np.array(predictions, dtype='int32').flatten()
    tokens = np.array(tokens, dtype='int32').flatten()
    labels = np.array(labels, dtype='int32').flatten()
    ambiguous_tokens = utils.identify_ambiguous_tokens(tokens.tolist(), labels.tolist(), id_to_word, word_to_id)

    accuracy = evaluate.load("accuracy")
    recall = evaluate.load("recall")
    precision = evaluate.load("precision")
    f1 = evaluate.load("f1")
    results_per_token = []

    for target_token in ambiguous_tokens:
        target_labels = np.array([label for token, label in zip(tokens, labels) if token == target_token])
        target_predictions = np.array([pred for token, pred in zip(tokens, predictions) if token == target_token])

        current_accuracy = accuracy.compute(predictions=target_predictions, references=target_labels)
        current_recall = recall.compute(predictions=target_predictions, references=target_labels, average=None, zero_division=False)["recall"]
        current_recall_sc = current_recall.tolist()[0]
        current_recall_sb = current_recall.tolist()[1]
        current_precision = precision.compute(predictions=target_predictions, references=target_labels, average=None, zero_division=False)["precision"]
        current_precision_sc = current_precision.tolist()[0]
        current_precision_sb = current_precision.tolist()[1]
        current_f1 = f1.compute(predictions=target_predictions, references=target_labels, average=None)["f1"]
        current_f1_sc = current_f1.tolist()[0]
        current_f1_sb = current_f1.tolist()[1]
        results_per_token.append((id_to_word[target_token], {"accuracy": current_accuracy['accuracy'],
                                                            "precision": [current_recall_sc, current_recall_sb],
                                                             "recall": [current_precision_sc, current_precision_sb],
                                                             "f1": [current_f1_sc, current_f1_sb]}))
    mean_accuracy = statistics.mean([float(item[1]["accuracy"]) for item in results_per_token])
    if name:
        out_file = f"{log_dir}/resultats_ambiguite_{name}.txt"
    else:
        out_file = f"{log_dir}/resultats_ambiguite.txt"
    print(f"Saving to {out_file}")
    with open(out_file, "w") as output_ambiguity:
        output_ambiguity.write(f"Mean accuracy: {mean_accuracy}.\n\n")
        for results in results_per_token:
            recall = ["Recall", results[1]["recall"][0], results[1]["recall"][1]]
            precision = ["Precision", results[1]["precision"][0], results[1]["precision"][1]]
            f1 = ["F1", results[1]["f1"][0], results[1]["f1"][1]]
            header = ["", "Segment Content", "Segment Boundary"]
            output_ambiguity.write(f"Results for {results[0]}: accuracy {results[1]['accuracy']}\n"
                  f"{utils.format_results(results=[precision, recall, f1], header=header, print_to_term=False)}"
                  f"\n\n\n")




def compute_metrics(predictions,
                    labels=None,
                    examples=None,
                    id_to_word=None,
                    last_epoch=False,
                    tokenizer=None,
                    bert_training=True,
                    mode="CharTokens",
                    log_file=None
                    ):
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

    predictions_as_probs = copy.deepcopy(predictions)
    if bert_training and labels is None:
        predictions, labels = predictions
        predictions = np.argmax(predictions, axis=2)
    elif bert_training and labels is not None:
        pass
    else:
        predictions = predictions.cpu()
        labels = labels.cpu()
        predictions = np.argmax(predictions, axis=2)

    # On teste un exemple pour voir si tout est OK.
    if last_epoch:
        if tokenizer:
            id_to_word = {ident: value for value, ident in tokenizer.get_vocab().items()}
        examples_number = 10
        random_number = random.randint(0, len(examples) - examples_number)
        example_range = range(random_number, random_number + examples_number)
        print(f"Showing example {random_number} to {random_number + examples_number}:")
        for idx in example_range:
            example = examples[idx].tolist()[1:]
            label = labels[idx].tolist()[1:]
            if mode != "CharTokens":
                # example_as_string = " ".join([id_to_word[ident] for ident in example]).replace(" ##", "")
                try:
                    position_first_left_padding = next(index for index, ident in enumerate(example) if ident == 0)
                except StopIteration:
                    position_first_left_padding = -1
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
            else:
                decoded_example = ["".join([id_to_word[char] for char in token if char not in [0, 1, 2, 3]]) for token in example]
                decoded_example = [item for item in decoded_example if item != ""]
                probs_no_padding = predictions_as_probs[idx].tolist()[1:]
                corresp_prediction = predictions[idx].tolist()[1:]
                corresp_prediction_as_classes = [item for item in corresp_prediction]
                corresp_label_as_classes = [item for item in label]
                correct = []
                for pred, label in zip(corresp_prediction_as_classes, corresp_label_as_classes):
                    if label == 1:
                        if pred == label:
                            correct.append("True")
                        else:
                            correct.append("False")
                    else:
                        if pred == 1:
                            correct.append("False")
                        else:
                            correct.append("")


                res = list(
                        zip(decoded_example,
                            corresp_prediction_as_classes,
                            corresp_label_as_classes,
                            correct,
                            probs_no_padding)
                )
                formatted = utils.format_results(results=res, header=["Token", "Prediction", "Target", "Correct", "Probability"],
                                     print_to_term=False)
                utils.append_to_file(formatted, log_file)



    # load the metrics we want to evaluate

    accuracy = evaluate.load("accuracy")
    recall = evaluate.load("recall")
    precision = evaluate.load("precision")
    f1 = evaluate.load("f1")

    # We flatten the 2 vectors to get a 1d vector of shape [num_examples*max_length]
    predictions = np.array(predictions, dtype='int32').flatten()
    labels = np.array(labels, dtype='int32').flatten()

    # On supprime le padding des données
    mask = labels != 2
    predictions = predictions[mask]
    labels = labels[mask]

    accuracy = accuracy.compute(predictions=predictions, references=labels)
    recall = recall.compute(predictions=predictions, references=labels, average=None)
    recall_l = []
    [recall_l.extend(v) for k, v in recall.items()]
    precision = precision.compute(predictions=predictions, references=labels, average=None)
    precision_l = []
    [precision_l.extend(v) for k, v in precision.items()]
    f1 = f1.compute(predictions=predictions, references=labels, average=None)
    f1_l = []
    [f1_l.extend(v) for k, v in f1.items()]

    results = {"accuracy": accuracy, "recall": recall_l, "precision": precision_l, "f1": f1_l}
    return results