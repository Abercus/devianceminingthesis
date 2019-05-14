"""
This is used to compare and extract rules from decision trees and RIPPER,
together with performance metrics being calculated


Requires: a python3 wrapper for Weka, numpy, sklearn, pandas
"""

import weka.core.jvm as jvm
from weka.classifiers import Classifier
from weka.core.converters import Loader

from weka.filters import Filter
from weka.classifiers import Evaluation         # to evaluate trained classifier
from weka.classifiers import PredictionOutput
import numpy as np
from sklearn.tree import DecisionTreeClassifier, _tree, export_graphviz
import pandas as pd

import traceback
from sklearn.tree._tree import TREE_LEAF


"""
- JRip vs DecisionTree (pruning + rule extraction)
- Comparisons between length of rules vs  accuracy 

"""
def read_df_csv(path):
    """
    Read csv from filepath into a pandas dataframe, comma separation.
    :param path:
    :return:
    """
    df = pd.read_csv(path, sep=",", na_filter=False)
    return df


def read_weka_csv(path, loader):
    """
    Read csv frm filepath into a data structure used by WEKA
    :param path:
    :param loader:
    :return:
    """
    jrip_data = loader.load_file(path)
    return jrip_data


def get_interesting(evaluation):
    """
    Extract metrics from a weka Evaluation class, which contains results of training a weka model
    :param evaluation:
    :return:
    """
    auc = evaluation.area_under_roc(1)
    f1 = evaluation.f_measure(1)
    precision = evaluation.precision(1)
    recall = evaluation.recall(1)
    accuracy = evaluation.percent_correct
    return [accuracy, auc, f1, precision, recall]


# Recursively backtrack to every leaf, extract decision rules

def find_rules(tree, features):
    """
    Finding rules from a decision tree
    :param tree:
    :param features:
    :return:
    """
    dt = tree.tree_
    feature_name = [
        features[i] if i != _tree.TREE_UNDEFINED else "undefined!"
        for i in dt.feature
    ]

    def visitor(node, depth, rules=None):
        indent = ' ' * depth
        if dt.feature[node] != _tree.TREE_UNDEFINED:
            #if len(rules) == 0:
            #    rules.append("if <{}> <= {}".format(indent, feature_name[node], round(dt.threshold[node], 2)))
            print('{}if <{}> <= {}:'.format(indent, feature_name[node], dt.threshold[node], 2))
            visitor(dt.children_left[node], depth + 1)
            print('{}else:'.format(indent))
            visitor(dt.children_right[node], depth + 1)
        else:
            #print("{}return {}".format(indent, np.argmax(dt.value[node][0])))
            print("{}return {}".format(indent, dt.value[node]))

    visitor(0, 1)

# Functions for pruning a tree.
# From https://stackoverflow.com/questions/51397109/prune-unnecessary-leaves-in-sklearn-decisiontreeclassifier
def is_leaf(inner_tree, index):
    # Check whether node is leaf node
    return (inner_tree.children_left[index] == TREE_LEAF and
            inner_tree.children_right[index] == TREE_LEAF)


def prune_index(inner_tree, decisions, index=0):
    # Start pruning from the bottom - if we start from the top, we might miss
    # nodes that become leaves during pruning.
    # Do not use this directly - use prune_duplicate_leaves instead.
    if not is_leaf(inner_tree, inner_tree.children_left[index]):
        prune_index(inner_tree, decisions, inner_tree.children_left[index])
    if not is_leaf(inner_tree, inner_tree.children_right[index]):
        prune_index(inner_tree, decisions, inner_tree.children_right[index])

    # Prune children if both children are leaves now and make the same decision:
    if (is_leaf(inner_tree, inner_tree.children_left[index]) and
        is_leaf(inner_tree, inner_tree.children_right[index]) and
        (decisions[index] == decisions[inner_tree.children_left[index]]) and
        (decisions[index] == decisions[inner_tree.children_right[index]])):
        # turn node into a leaf by "unlinking" its children
        inner_tree.children_left[index] = TREE_LEAF
        inner_tree.children_right[index] = TREE_LEAF


def prune_duplicate_leaves(mdl):
    # Remove leaves if both
    decisions = mdl.tree_.value.argmax(axis=2).flatten().tolist() # Decision for each node
    prune_index(mdl.tree_, decisions)


def run_weka_csv_train_test(train_file_path, test_file_path):
    """
    1) From previous process, for each fold create input .csv's which will then be read here
    1.1) Input csv will be on exact same data, which was fed to DT model
    2) Extract the rules using JRip
    3) Evaluate predictions with same metrics as was done for previous work
    3.1) Accuracy, AUC, F-Score, Precision, Recall
    need to make new .csv, which contains both, payload and usual stuff....
    :return:
    """
    train_df = read_df_csv(train_file_path)
    test_df = read_df_csv(test_file_path)

    cls = Classifier(classname="weka.classifiers.rules.JRip") #options=["-O", "2"]), default opt. is 2

    loader = Loader(classname="weka.core.converters.CSVLoader")
    # print(cls.to_help())
    train_jrip_data = read_weka_csv(train_file_path, loader)
    test_jrip_data = read_weka_csv(test_file_path, loader)

    # If dataset included Case ID
    #train_case_id = train_jrip_data.attribute_by_name("Case_ID")
    #test_case_id = test_jrip_data.attribute_by_name("Case_ID")
    #test_jrip_data.delete_attribute(test_case_id.index)
    #train_jrip_data.delete_attribute(train_case_id.index)

    train_label_attribute = train_jrip_data.attribute_by_name("Label")
    test_label_attribute = test_jrip_data.attribute_by_name("Label")

    # Convert numeric attribut to nominal. Required for label!
    nominal = Filter(classname="weka.filters.unsupervised.attribute.NumericToNominal", options=["-R", "last"])
    nominal.inputformat(train_jrip_data)
    nominaldata1 = nominal.filter(train_jrip_data)

    nominaldata1.class_index = train_label_attribute.index
    nominaldata2 = nominal.filter(test_jrip_data)  # re-use the initialized filter!

    nominaldata2.class_index = test_label_attribute.index
    msg = nominaldata1.equal_headers(nominaldata2)

    if msg is not None:
        raise Exception("Train and test not compatible:\n" + msg)

    # Build classifier
    cls.build_classifier(nominaldata1)
    # Get rules
    print(cls.jwrapper)


    pred_output = PredictionOutput(classname="weka.classifiers.evaluation.output.prediction.PlainText",
                                   options=["-distribution"])

    # Perform evaluation on train and test data
    evl = Evaluation(nominaldata1)
    evl.test_model(cls, nominaldata2, output=pred_output)

    evl2 = Evaluation(nominaldata1)
    evl2.test_model(cls, nominaldata1, output=pred_output)

    # Return interesting metrics for both train and test.
    return get_interesting(evl2), get_interesting(evl)


def get_code(tree, feature_names):
    """Modified to print decision rules instead

    Args
    ----
    tree -- scikit-leant DescisionTree.
    feature_names -- list of feature names.
    target_names -- list of target (class) names.
    spacer_base -- used for spacing code (default: "    ").

    Notes
    -----
    based on http://stackoverflow.com/a/30104792.
    """
    left      = tree.tree_.children_left
    right     = tree.tree_.children_right
    threshold = tree.tree_.threshold
    features  = [feature_names[i] for i in tree.tree_.feature]
    value = tree.tree_.value

    global rules_count
    rules_count = 0

    def recurse(left, right, threshold, features, node, depth, current_text=""):
        global rules_count
        if (threshold[node] != -2):

            if left[node] != -1:
                if current_text != "":
                    pass_text = current_text +  " and (" + features[node] + " <= " + str(threshold[node]) + ")"
                else:
                    pass_text = current_text +  "(" + features[node] + " <= " + str(threshold[node]) + ")"
                recurse(left, right, threshold, features,
                        left[node], depth+1, pass_text)

            if right[node] != -1:
                if current_text != "":
                    pass_text = current_text + " and (" + features[node] + " > " + str(threshold[node]) + ")"
                else:
                    pass_text = current_text + "(" + features[node] + " > " + str(threshold[node]) + ")"
                recurse(left, right, threshold, features,
                        right[node], depth+1, pass_text)
        else:
            target = value[node]
            pos_samples = target[0][1]
            neg_samples = target[0][0]

            if pos_samples > neg_samples:
                current_text += " => Label=1" + " (" + str(pos_samples) + "/" + str(neg_samples) + ")"
                print(current_text)
                rules_count += 1

    recurse(left, right, threshold, features, 0, 0)

    return rules_count

def python_run():
    """
    Run python model (decision tree) on already ready encoded output from general pipeline.
    Output for other program is in folder snapshots/, in where train and test .csv's are
    """

    coverages = [5, 15, 25]
    splits = [1, 2, 3, 4, 5]


    # Encodings used, either for pure data or w/o
    #"""
    encodings = [
        "payload",
        "baseline_payload",
        "declare_data",
        "sequence_data_tr",
        "sequence_data_tra",
        "sequence_data_mr",
        "sequence_data_mra",
        "hybrid_data",
    ]
    #"""
    """
    encodings = [
        #"payload",
        "baseline",
        "declare",
        "sequence_tr",
        "sequence_tra",
        "sequence_mr",
        "sequence_mra",
        "hybrid",
    ]
    """

    # Experiment used.
    #experiment = "sepsis"
    experiment = "bpi2011_cc"
    #experiment = "bpi2011_m16"
    #experiment = "bpi2011_m13"
    #experiment = "bpi2011_t101"
    #experiment = "xray"
    #experiment = "synthmr"
    #experiment = "synthmra"
    ress = {}

    # Go over all coverage and encodings and extract rules.
    for coverage in coverages:
        for encoding in encodings:
            print("Working on {} with {} and {}".format(experiment, coverage, encoding))
            rules_counts = []
            for split in splits:
                print("Split {}".format(split))
                train_name = "snapshots/train_{}_{}_{}_{}.csv".format(experiment, encoding, coverage, split)
                test_name = "snapshots/test_{}_{}_{}_{}.csv".format(experiment, encoding, coverage, split)

                train_df = pd.read_csv(train_name)
                y_train = train_df.pop("Label")
                feature_names = train_df.columns

                clf = DecisionTreeClassifier(max_depth=10, min_samples_leaf=5)
                clf.fit(train_df.values, y_train)
                prune_duplicate_leaves(clf)

                rules_count = get_code(clf, feature_names)
                rules_counts.append(rules_count)
                #test_df = pd.read_csv(test_name)
                #y_test = test_df["Label"]

            avg_rules = np.mean(rules_counts)
            std_rules = np.std(rules_counts)
            ress[(encoding, coverage)] = (avg_rules, std_rules)
    for coverage in coverages:
        for encoding in encodings:
            k = ress[(encoding, coverage)]
            print(k[0], k[1])
        print("")


def testcase():
    coverages = [5, 15, 25]
    splits = [1,2,3,4,5]
    #"""
    encodings = [
        "payload",
        "baseline_payload",
        "declare_data",
        "sequence_data_tr",
        "sequence_data_tra",
        "sequence_data_mr",
        "sequence_data_mra",
        "hybrid_data",
    ]
    #"""
    """
    encodings = [
        #"payload",
        "baseline",
        "declare",
        "sequence_tr",
        "sequence_tra",
        "sequence_mr",
        "sequence_mra",
        "hybrid",
    ]
    """

    experiments = ["bpi2011_cc", "bpi2011_m13", "bpi2011_m16", "bpi2011_t101"]
    for experiment in experiments:
        print("Started with experiment {}".format(experiment))
        ttrain_results = []
        ttest_results = []
        for coverage in coverages:
            for encoding in encodings:
                print("Working on {} with  {} and {}".format(experiment, coverage, encoding))
                train_accuracies = []
                train_precisions = []
                train_rcs = []
                train_f1s = []
                train_aucs = []

                test_accuracies = []
                test_precisions = []
                test_f1s = []
                test_rcs = []
                test_aucs = []
                for split in splits:
                    train_name = "snapshots/train_{}_{}_{}_{}.csv".format(experiment, encoding, coverage, split)
                    test_name = "snapshots/test_{}_{}_{}_{}.csv".format(experiment, encoding, coverage, split)

                    train_results, test_results = run_weka_csv_train_test(train_name, test_name)

                    #for i in range(len(train_results)):
                    #    if math.isnan(train_results[i]):
                    #        train_results[i] = 0

                    #for i in range(len(test_results)):
                    #    if math.isnan(test_results[i]):
                    #        test_results[i] = 0

                    test_accuracy, test_auc, test_f1, test_precision, test_rc = test_results
                    train_accuracy, train_auc, train_f1, train_precision, train_rc = train_results

                    train_accuracies.append(train_accuracy)
                    train_precisions.append(train_precision)
                    train_rcs.append(train_rc)
                    train_f1s.append(train_f1)
                    train_aucs.append(train_auc)

                    test_accuracies.append(test_accuracy)
                    test_precisions.append(test_precision)
                    test_rcs.append(test_rc)
                    test_f1s.append(test_f1)
                    test_aucs.append(test_auc)

                   #print(test_accuracy, test_auc, test_f1, test_rc, test_precision)

                #print(coverage, encoding)
                ttest_results.append("{},{},{},{},{},{},{},{},{},{}".format(
                    np.mean(test_accuracies), np.std(test_accuracies),
                    np.mean(test_aucs), np.std(test_aucs),
                    np.mean(test_f1s), np.std(test_f1s),
                    np.mean(test_rcs), np.std(test_rcs),
                    np.mean(test_precisions), np.std(test_precisions)))

                ttrain_results.append("{},{},{},{},{},{},{},{},{},{}".format(
                    np.mean(train_accuracies), np.std(train_accuracies),
                    np.mean(train_aucs), np.std(train_aucs),
                    np.mean(train_f1s), np.std(train_f1s),
                    np.mean(train_rcs), np.std(train_rcs),
                    np.mean(train_precisions), np.std(train_precisions)))


        print("Test results")
        for i in range(len(coverages)):
            print("")
            for j in range(len(encodings)):
                print(ttest_results[i*len(coverages)+j])

        print("Train results")
        for i in range(len(coverages)):
            print("")
            for j in range(len(encodings)):
                print(ttrain_results[i * len(coverages) + j])


#RUN = "weka"
RUN = "python" # Either run weka or python code
if __name__ == "__main__":
    if RUN == "python":
        python_run()

    elif RUN == "weka":
        try:
            # Weka wrapper uses JVM
            jvm.start()
            testcase()
        except Exception as e:
            print(traceback.format_exc())
        finally:
            jvm.stop()

