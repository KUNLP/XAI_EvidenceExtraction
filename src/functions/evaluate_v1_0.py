from __future__ import print_function
from collections import Counter
import string
import re
import argparse
import json
import sys
import os
def normalize_answer(s):
    def remove_(text):
        ''' 불필요한 기호 제거 '''
        text = re.sub("'", " ", text)
        text = re.sub('"', " ", text)
        text = re.sub('《', " ", text)
        text = re.sub('》', " ", text)
        text = re.sub('<', " ", text)
        text = re.sub('>', " ", text)
        text = re.sub('〈', " ", text)
        text = re.sub('〉', " ", text)
        text = re.sub("\(", " ", text)
        text = re.sub("\)", " ", text)
        text = re.sub("‘", " ", text)
        text = re.sub("’", " ", text)
        return text

    def white_space_fix(text):
        return ' '.join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return ''.join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_punc(lower(remove_(s))))


def f1_score(prediction, ground_truth):
    prediction_tokens = normalize_answer(prediction).split()
    ground_truth_tokens = normalize_answer(ground_truth).split()

    # F1 by character
    prediction_Char = []
    for tok in prediction_tokens:
        now = [a for a in tok]
        prediction_Char.extend(now)
    ground_truth_Char = []
    for tok in ground_truth_tokens:
        now = [a for a in tok]
        ground_truth_Char.extend(now)
    common = Counter(prediction_Char) & Counter(ground_truth_Char)
    num_same = sum(common.values())
    if num_same == 0:
        return 0

    precision = 1.0 * num_same / len(prediction_Char)
    recall = 1.0 * num_same / len(ground_truth_Char)
    f1 = (2 * precision * recall) / (precision + recall)

    return f1

def exact_match_score(prediction, ground_truth):
    return (normalize_answer(prediction) == normalize_answer(ground_truth))


def metric_max_over_ground_truths(metric_fn, prediction, ground_truths):
    scores_for_ground_truths = []
    for ground_truth in ground_truths:
        score = metric_fn(prediction, ground_truth)
        scores_for_ground_truths.append(score)
    return max(scores_for_ground_truths)


def evaluate(dataset, predictions):
    f1 = exact_match = total = 0
    for article in dataset:
        title = article["_id"]
        q_id = article['_id']
        if q_id not in predictions:
            message = 'Unanswered question ' + title + \
                      ' will receive score 0.'
            print(message, file=sys.stderr)
            continue
        total += 1
        ground_truths = [article["answer"]]
        prediction = predictions[q_id]
        e = metric_max_over_ground_truths(
            exact_match_score, prediction, ground_truths)
        exact_match += metric_max_over_ground_truths(
            exact_match_score, prediction, ground_truths)
        f1 += metric_max_over_ground_truths(
            f1_score, prediction, ground_truths)

    exact_match = 100.0 * exact_match / total
    f1 = 100.0 * f1 / total
    return {'official_exact_match': exact_match, 'official_f1': f1}


def eval_during_train(args, global_step):
    expected_version = 'KorQuAD_v1.0'

    dataset_file = os.path.join(args.data_dir, args.predict_file)
    prediction_file = os.path.join(args.output_dir, 'predictions_{}.json'.format(global_step))

    with open(dataset_file) as dataset_f:
        dataset_json = json.load(dataset_f)

        dataset = dataset_json
    with open(prediction_file) as prediction_f:
        predictions = json.load(prediction_f)

    return evaluate(dataset, predictions)


if __name__ == '__main__':
    expected_version = 'KorQuAD_v1.0'
    parser = argparse.ArgumentParser(
        description='Evaluation for KorQuAD ' + expected_version)
    parser.add_argument('--dataset_file', default="../../data/hotpot_dev_distractor_v1.json")
    parser.add_argument('--prediction_file',default="../../proposed_model_1019/predictions_14000.json")

    args = parser.parse_args()
    with open(args.dataset_file) as dataset_file:
        dataset_json = json.load(dataset_file)
        # read_version = "_".join(dataset_json['version'].split("_")[:-1])
        # if (read_version != expected_version):
        #     print('Evaluation expects ' + expected_version +
        #           ', but got dataset with ' + read_version,
        #           file=sys.stderr)
        dataset = dataset_json
    with open(args.prediction_file) as prediction_file:
        predictions = json.load(prediction_file)
    print(json.dumps(evaluate(dataset, predictions)))
