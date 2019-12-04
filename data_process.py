import csv
import demjson
import numpy as np
import random
import pandas as pd
from sklearn.model_selection import train_test_split


def read_sql_file(dataset_path):
    with open(dataset_path, 'rb') as csv_file:
        with open('child_data.csv', 'w') as data_file:
            writer = csv.writer(data_file)
            writer.writerow(['child_id', 'activity_id', 'score'])
            spam_reader = csv.DictReader(csv_file)
            for row in spam_reader:
                child_id = demjson.decode(row['\xef\xbb\xbf_col4'])['childId']
                score = demjson.decode(row['\xef\xbb\xbf_col4'])['scoreProficiencyEvent']['entityScores'][0]['score']
                activity_id = demjson.decode(row['\xef\xbb\xbf_col4'])['scoreProficiencyEvent']['activityIds'][0]
                writer.writerow([child_id, activity_id, score])

def generate_dict(dataset_path):
    data_df = pd.read_csv(dataset_path, header=None, names=["child_id", "activity_id", "kp_label", "score"], low_memory=False)
    aid_int_activity = pd.DataFrame({"activity_id": data_df["activity_id"].unique(), "int_id": range(len(data_df["activity_id"].unique()))})
    aid_int_map_activity = aid_int_activity.set_index("activity_id").to_dict()["int_id"]
    del aid_int_map_activity['activity_id']
    aid_int_map_activity['placeholder'] = 0
    aid_int_kp = pd.DataFrame({"kp_label": data_df["kp_label"].unique(), "int_id": range(len(data_df["kp_label"].unique()))})
    aid_int_map_kp = aid_int_kp.set_index("kp_label").to_dict()["int_id"]
    del aid_int_map_kp['kp_label']
    aid_int_map_kp['placeholder'] = 0
    return aid_int_map_activity, aid_int_map_kp

def read_file(dataset_path, aid_int_map_activity, aid_int_map_kp):
    seqs_by_student = {}
    num_activities = 0
    num_kp = 0
    with open(dataset_path, 'r') as f:
        for num, line in enumerate(f):
            if num == 0:
                continue
            fields = line.strip().split(",")
            if float(fields[3]) < 0.2:
                continue
            student, problem, kp, is_correct = fields[0], aid_int_map_activity[fields[1]], aid_int_map_kp[fields[2]], float(fields[3])
            num_activities = max(num_activities, problem)
            num_kp = max(num_kp, kp)
            seqs_by_student[student] = seqs_by_student.get(student, []) + [[problem, kp, is_correct]]
    return seqs_by_student, num_activities + 1, num_kp + 1

def random_key(key_file, seqs_by_student):
    sorted_keys = sorted(seqs_by_student.keys())
    random.seed(1)
    test_keys = set(random.sample(sorted_keys, int(len(sorted_keys) * 0.2)))
    with open(key_file, 'w') as f:
        for num, key in enumerate(test_keys):
            if num == len(test_keys) - 1:
                f.write(str(key))
                continue
            f.write(str(key)+',')

def split_test_dataset(key_file, seqs_by_student):
    with open(key_file, 'r') as f:
        for key in f.readlines():
            test_keys = key.split(',')
    test_seqs = [seqs_by_student[k] for k in seqs_by_student if k in test_keys]
    train_seqs = [seqs_by_student[k] for k in seqs_by_student if k not in test_keys]
    return train_seqs, test_seqs

def split_validate_dataset(data, seq_len):
    print data.shape
    x_train, x_validate, y_train, y_validate = train_test_split(zip(data[:, :, :-1], seq_len), data[:, :, 2], test_size=0.2, random_state=0)
    return x_train, y_train, x_validate, y_validate

def format_data(seqs, max_seq_len):
    data = []
    seq_len = []
    for child_record in seqs:
        if len(child_record) < 30:
            continue
        while len(child_record) > max_seq_len:
            data.append(child_record[:max_seq_len])
            seq_len.append(max_seq_len - 1)
            child_record = child_record[max_seq_len:]
        data.append(child_record + [[0, 0, 0]] * (max_seq_len - len(child_record)))
        seq_len.append(len(child_record) - 1)
    return np.array(data), np.array(seq_len)

def tansform_input(input, pre_sorce, num_activities):
    for batch in range(input.shape[0]):
        for activity in range(input.shape[1]):
            if pre_sorce[batch][activity] > 0.2 and pre_sorce[batch][activity] <= 0.4:
                input[batch][activity] = input[batch][activity] + 0 * (num_activities - 1)
            elif pre_sorce[batch][activity] > 0.4 and pre_sorce[batch][activity] <= 0.6:
                input[batch][activity] = input[batch][activity] + 1 * (num_activities - 1)
            elif pre_sorce[batch][activity] > 0.6 and pre_sorce[batch][activity] <= 0.8:
                input[batch][activity] = input[batch][activity] + 2 * (num_activities - 1)
            elif pre_sorce[batch][activity] > 0.8 and pre_sorce[batch][activity] <= 1:
                input[batch][activity] = input[batch][activity] + 3 * (num_activities - 1)
    return input
















