from Utils.CoQAUtils import save_errors
import json
import numpy as np
import pandas as pd

with open('../coqa/conf~/run_11/prediction.json') as json_file:
    pred_json = json.load(json_file)

with open('../coqa/conf~/run_11/score_per_instance.json') as json_file:
    score_json = json.load(json_file)

total_fail, partial_fail = save_errors(pred_json, score_json)

#test = ensemble_predict(pred_list,score_list)
print("{} total fail, {} partial fail et {} perfect good".
      format(len(total_fail), len(partial_fail), len(pred_json) - len(total_fail) - len(partial_fail)))
#print(1-display_error(pred_json, score_json)[0]/len(pred_json)Merci Vincent

with open('../coqa/data/coqa-dev-v1.0.json') as json_file:
    coqa_json = json.load(json_file)

errors = []
total_fail_df = pd.DataFrame(total_fail)
for qst in coqa_json['data']:
    df2 = total_fail_df[(total_fail_df['id'] == qst['id'])]
    story = qst['story']
    for i in range(len(qst['questions'])):
        for j in range(len(df2.turn_id.values)):
            turn_id = df2.turn_id.values[j]
            if qst['questions'][i]['turn_id'] == turn_id:
                question = qst['questions'][i]['input_text']
                truth_answer = qst['answers'][i]['input_text']
                predicted_answer = df2.answer.values[j]
                errors.append({'question':question,
                               'truth_answer': truth_answer,
                               'predicted_answer': predicted_answer,
                               'turn_id' : turn_id,
                               'story': story})

part_errors = []
partial_fail_df = pd.DataFrame(partial_fail)
for qst in coqa_json['data']:
    df2 = partial_fail_df[(partial_fail_df['id'] == qst['id'])]
    story = qst['story']
    for i in range(len(qst['questions'])):
        for j in range(len(df2.turn_id.values)):
            turn_id = df2.turn_id.values[j]
            if qst['questions'][i]['turn_id'] == turn_id:
                question = qst['questions'][i]['input_text']
                truth_answer = qst['answers'][i]['input_text']
                predicted_answer = df2.answer.values[j]
                part_errors.append({'question':question,
                               'truth_answer': truth_answer,
                               'predicted_answer': predicted_answer,
                               'turn_id' : turn_id,
                               'story': story})


import csv

with open('../coqa/total_fail.csv', 'w', newline='') as csvfile:
    fieldnames = errors[0].keys()
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

    writer.writeheader()
    writer.writerows(errors)

with open('../coqa/partial_fail.csv', 'w', newline='') as csvfile:
    fieldnames = errors[0].keys()
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

    writer.writeheader()
    writer.writerows(errors)

print('test')
