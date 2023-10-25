import json
import os


predict_file = open('./proposed_model_1020/nbest_predictions_20000.json','r',encoding='utf8')
answer_file = open('./data/hotpot_dev_distractor_v1.json','r',encoding='utf8')
predictions = json.load(predict_file)
answers = json.load(answer_file)

prediction_dict = {}
n = 0
c = 0
for qas in predictions.keys():
    # if len(predictions[qas][0]["evidence"])>1:
    #     print(qas)
    prediction_dict[qas] = [predictions[qas][0]["text"], predictions[qas][0]["evidence"]]


# print(c/n)

all_answer_num = 0
all_predict_num = 0
all_correct = 0

precision_list = []
recall_list = []
f1_list = []
for data in answers:
    answer_num = 0
    predict_num = 0
    correct = 0
    qas_id = data["_id"]
    answer_text = data["answer"]
    documents = {e[0]: e[1] for e in data["context"]}
    supporting_facts = data['supporting_facts']
    supporting_sentences = []
    for idx, support_fact in enumerate(supporting_facts):
        sentence = documents[support_fact[0]][support_fact[1]]
        supporting_sentences.append(sentence)
    # if answer_text.strip() != prediction_dict[qas_id][0].strip():
    #     continue
    # if len(prediction_dict[qas_id][1]) < 3:
    #     continue

    prediction = list(set(prediction_dict[qas_id][1]))
    # tmp = ''.join(prediction)
    # if answer_text not in tmp:
    #     print("??")
    for sent in supporting_sentences:
        if sent in prediction:
            correct +=1

            all_correct +=1

        answer_num +=1

    predict_num += len(prediction)
    all_answer_num += len(supporting_sentences)
    all_predict_num += len(prediction)
    if not predict_num:
        predict_num = 1e-10
    precision = correct / predict_num
    recall = correct / answer_num
    f1 = (2*precision*recall) / (recall + precision + 1e-10)

    precision_list.append(precision)
    recall_list.append(recall)
    f1_list.append(f1)

per_precision = sum(precision_list) / len(precision_list)
per_recall = sum(recall_list) / len(recall_list)
per_f1 = sum(f1_list) / len(f1_list)


print("Per Precision : {}\tRecall : {}\tF1 : {}".format(round(per_precision, 3), round(per_recall, 3), round(per_f1, 3)))

all_precision = all_correct / all_predict_num
all_recall = all_correct / all_answer_num
all_f1 = (2*all_precision*all_recall) / (all_recall + all_precision + 1e-10)

print("All Precision : {}\tRecall : {}\tF1 : {}".format(round(all_precision, 3), round(all_recall, 3), round(all_f1, 3)))




# 다음주 화요일 논의

