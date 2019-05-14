"""
Code to find and encode IA encoding (Individual Activities) or baseline

"""

from deviancecommon import read_XES_log, xes_to_positional, extract_unique_events_transformed
import numpy as np
from declaredevmining import split_log_train_test
import pandas as pd

import os, shutil



def transform_log(train_log, activity_set):
    train_names = []

    train_labels = []
    train_data = []
    for trace in train_log:
        name = trace["name"]
        label = trace["label"]
        res = []
        train_labels.append(label)
        train_names.append(name)
        for event in activity_set:
            if event in trace["events"]:
                res.append(len(trace["events"][event]))
            else:
                res.append(0)

        train_data.append(res)

    np_train_data = np.array(train_data)

    train_df = pd.DataFrame(np_train_data)
    train_df.columns = activity_set
    train_df["Case_ID"] = train_names
    train_df["Label"] = train_labels


    train_df.set_index("Case_ID")

    return train_df


def baseline(inp_folder, logPath):

    log = read_XES_log(logPath)

    transformed_log = xes_to_positional(log)

    train_log, test_log = split_log_train_test(transformed_log, 0.8)
    # Collect all different IA's

    activitySet = list(extract_unique_events_transformed(train_log))
    # Transform to matrix

    print("Train data")
    # train data

    train_df = transform_log(train_log, activitySet)

    print("Test data")
    # test data
    test_df = transform_log(test_log, activitySet)

    train_df.to_csv(inp_folder + "/baseline_train.csv", index=False)

    test_df.to_csv(inp_folder + "/baseline_test.csv", index=False)


def move_baseline_files(inp_folder, output_folder, split_nr):
    source = inp_folder # './baselineOutput/'
    dest1 = './' + output_folder + '/split' + str(split_nr) + "/base/"
    files = os.listdir(source
                       )
    for f in files:
        shutil.move(source + f, dest1)


def run_baseline(experiment_name, log_path, results_folder):
    for logNr in range(5):
        logPath = log_path.format(logNr + 1)
        folder_name = "./{}_baseline/".format(experiment_name)
        if not os.path.exists(folder_name):
            os.makedirs(folder_name)

        baseline(folder_name, logPath)
        move_baseline_files(folder_name, results_folder, logNr + 1)


