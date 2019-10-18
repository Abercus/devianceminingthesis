"""
Main file for deviance mining
"""
from random import shuffle
import numpy as np
from declaretemplates_new import *
#from declaretemplates import *
from deviancecommon import *
import pandas as pd

import shutil
import os



def reencode_map(val):
    if val == -1:
        return "violation"
    elif val == 0:
        return "vacuous"
    elif val == 1:
        return "single"
    elif val == 2:
        return "multi"


def reencode_declare_results(train_df, test_df):
    """
    Given declare results dataframe, reencode the results such that they are one-hot encodable
    If Frequency is -1, it means that there was a violation, therefore it will be one class
    If Frequency is 0, it means that the constraint was vacuously filled, it will be second class
    If Frequency is 1, then it will be class of single activation
    If Frequency is 2... then it will be a class of multiple activation

    In total there will be 4 classes
    :param train_df:
    :param test_df:
    :return:
    """

    train_size = len(train_df)

    union = pd.concat([train_df, test_df], sort=False)

    # First, change all where > 2 to 2.
    union[union > 2] = 2
    # All -1's to "VIOLATION"
    union.replace({
        -1: "violation",
        0: "vacuous",
        1: "single",
        2: "multi"
    }, inplace=True)

    union = pd.get_dummies(data=union, columns=train_df.columns)
    # Put together and get_dummies for one-encoded features

    train_df = union.iloc[:train_size, :]
    test_df = union.iloc[train_size:, :]

    return train_df, test_df


def apply_template_to_log(template, candidate, log):
    results = []
    for trace in log:
        result, vacuity = apply_template(template, trace, candidate)

        results.append(result)

    return results


def generate_candidate_constraints(candidates, templates, train_log, constraint_support=None):
    all_results = {}

    for template in templates:
        print("Started working on {}".format(template))
        for candidate in candidates:
            if len(candidate) == template_sizes[template]:
                constraint_result = apply_template_to_log(template, candidate, train_log)

                if constraint_support:
                    satisfaction_count = len([v for v in constraint_result if v != 0])
                    if satisfaction_count >= constraint_support:
                        all_results[template + ":" + str(candidate)] = constraint_result

                else:
                    all_results[template + ":" + str(candidate)] = constraint_result

    return all_results


def find_if_satisfied_by_class(constraint_result, log, support_norm, support_dev):
    fulfill_norm = 0
    fulfill_dev = 0
    for i, trace in enumerate(log):
        ## TODO: Find if it is better to have > 0 or != 0.
        if constraint_result[i] > 0:
        #if constraint_result[i] != 0:
            if trace["label"] == 1:
                fulfill_dev += 1
            else:
                fulfill_norm += 1

    norm_pass = fulfill_norm >= support_norm
    dev_pass = fulfill_dev >= support_dev

    return norm_pass, dev_pass


def generate_train_candidate_constraints(candidates, templates, train_log, constraint_support_norm,
                                         constraint_support_dev, filter_t=True):
    all_results = {}
    for template in templates:
        print("Started working on {}".format(template))
        for candidate in candidates:
            if len(candidate) == template_sizes[template]:
                candidate_name = template + ":" + str(candidate)
                constraint_result = apply_template_to_log(template, candidate, train_log)
                satis_normal, satis_deviant = find_if_satisfied_by_class(constraint_result, train_log,
                                                                         constraint_support_norm,
                                                                         constraint_support_dev)

                if not filter_t or (satis_normal or satis_deviant):
                    all_results[candidate_name] = constraint_result

    return all_results


def generate_test_candidate_constraints(candidates, templates, test_log, train_results):
    all_results = {}
    for template in templates:
        print("Started working on {}".format(template))
        for candidate in candidates:
            if len(candidate) == template_sizes[template]:
                candidate_name = template + ":" + str(candidate)
                if candidate_name in train_results:
                    constraint_result = apply_template_to_log(template, candidate, test_log)

                    all_results[candidate_name] = constraint_result

    return all_results


def transform_results_to_numpy(results, train_log):
    """
    Transforms results structure into numpy arrays
    :param results:
    :param train_log:
    :return:
    """
    labels = [trace["label"] for trace in train_log]
    trace_names = [trace["name"] for trace in train_log]
    matrix = []
    featurenames = []

    for feature, result in results.items():
        matrix.append(result)
        featurenames.append(feature)

    nparray_data = np.array(matrix).T
    nparray_labels = np.array(labels)
    nparray_names = np.array(trace_names)
    return nparray_data, nparray_labels, featurenames, nparray_names


def filter_candidates_by_support(candidates, log, support_norm, support_dev):
    filtered_candidates = []
    for candidate in candidates:
        count_dev = 0
        count_norm = 0
        for trace in log:
            ev_ct = 0
            for event in candidate:
                if event in trace["events"]:
                    ev_ct += 1
                else:
                    break
            if ev_ct == len(candidate):  # all candidate events in trace
                if trace["label"] == 1:
                    count_dev += 1
                else:
                    count_norm += 1

            if count_dev >= support_dev or count_norm >= support_norm:
                filtered_candidates.append(candidate)
                break

    return filtered_candidates


def count_classes(log):
    deviant = 0
    normal = 0
    for trace in log:
        if trace["label"] == 1:
            deviant += 1
        else:
            normal += 1

    return normal, deviant



def declare_deviance_mining(log, templates=None, to_shuffle=False, filter_t=True, reencode=False):
    print("Filter_t", filter_t)
    if not templates:
        templates = template_sizes.keys()

    constraint_threshold = 0.1
    candidate_threshold = 0.1

    # Read into suitable data structure
    transformed_log = xes_to_positional(log)
    if to_shuffle:
        shuffle(transformed_log)

    train_log, test_log = split_log_train_test(transformed_log, 0.8)

    # Extract unique activities from log
    events_set = extract_unique_events_transformed(train_log)

    # Brute force all possible candidates
    candidates = [(event,) for event in events_set] + [(e1, e2) for e1 in events_set for e2 in events_set if e1 != e2]
    print("Start candidates:", len(candidates))

    # Count by class
    normal_count, deviant_count = count_classes(train_log)
    print("{} deviant and {} normal traces in train set".format(deviant_count, normal_count))
    ev_support_norm = int(normal_count * candidate_threshold)
    ev_support_dev = int(deviant_count * candidate_threshold)

    if filter_t:
        print(filter_t)
        print("Filtering candidates by support")
        candidates = filter_candidates_by_support(candidates, train_log, ev_support_norm, ev_support_dev)
        print("Support filtered candidates:", len(candidates))

    constraint_support_dev = int(deviant_count * constraint_threshold)
    constraint_support_norm = int(normal_count * constraint_threshold)

    train_results = generate_train_candidate_constraints(candidates, templates, train_log, constraint_support_norm,
                                                         constraint_support_dev, filter_t=filter_t)

    test_results = generate_test_candidate_constraints(candidates, templates, test_log, train_results)
    print("Candidate constraints generated")

    # transform to numpy
    # get trace names
    train_data, train_labels, featurenames, train_names = transform_results_to_numpy(train_results, train_log)
    test_data, test_labels, _, test_names = transform_results_to_numpy(test_results, test_log)

    train_df = pd.DataFrame(train_data, columns=featurenames)
    test_df = pd.DataFrame(test_data, columns=featurenames)

    # Reencoding data
    if reencode:
        print("Reencoding data")
        train_df, test_df = reencode_declare_results(train_df, test_df)

    train_df["Case_ID"] = train_names
    train_df["Label"] = train_labels.tolist()
    test_df["Case_ID"] = test_names
    test_df["Label"] = test_labels.tolist()
    train_df.to_csv("declareOutput/declare_train.csv", index=False)
    test_df.to_csv("declareOutput/declare_test.csv", index=False)


def run_deviance_new(log_path, results_folder, templates=None, filter_t=True, reencode=False):
    for logNr in range(5):
        args = {
            "logPath": log_path.format(logNr + 1),
            "labelled": True
        }
        
        folder_name ="./declareOutput/"
        if not os.path.exists(folder_name):
            os.makedirs(folder_name)

        print("Deviance mining filtering:", filter_t)
        
        
        deviance_main(args, templates=templates, filter_t=filter_t, reencode=reencode)

        move_out_files_new(logNr + 1, results_folder)


def move_out_files_new(splitNr, results_folder):
    source = './declareOutput/'
    dest1 = './' + results_folder + '/split' + str(splitNr) + "/declare/"

    files = os.listdir(source)

    for f in files:
        shutil.move(source + f, dest1)


def deviance_main(args, templates=None, filter_t=True, reencode=False):
    print("Working on: " + args["logPath"], "Filtering:", filter_t)
    log = read_XES_log(args["logPath"])
    declare_deviance_mining(log, templates=templates, filter_t=filter_t, reencode=reencode)


if __name__ == "__main__":
    log_path = "logs/sepsis_tagged_er.xes"
    args = {
        "logPath": log_path,
        "labelled": True
    }
    deviance_main(args, reencode=True)
