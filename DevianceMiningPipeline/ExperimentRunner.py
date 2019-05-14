"""
Full pipeline for deviance mining experiments

Author: Joonas Puura
"""

from random import shuffle
import os

import pandas as pd
import numpy as np

import shutil

from sklearn.metrics import roc_auc_score, accuracy_score, precision_score, recall_score, f1_score
from sklearn.tree import DecisionTreeClassifier, _tree, export_graphviz
from sklearn.feature_selection import SelectKBest, chi2

from opyenxes.data_out.XesXmlSerializer import XesXmlSerializer
from opyenxes.factory.XFactory import XFactory

from deviancecommon import read_XES_log
from baseline_runner import run_baseline
from declaredevmining import run_deviance_new
from sequence_runner import run_sequences

from ddm_newmethod_fixed_new import run_declare_with_data

from sklearn.preprocessing import StandardScaler

## Remove features ...
from skfeature.function.similarity_based import fisher_score
from sklearn.decomposition import PCA
from collections import defaultdict


from payload_extractor import run_payload_extractor

import arff


def fisher_calculation(X, y):
    # Calculates fisher score
    """
    Calculate fisher score
    https://papers.nips.cc/paper/2909-laplacian-score-for-feature-selection.pdf
    :param data:
    :return:
    """

    #print(X[0,:])
    # Find mean and variance for full dataset

    #for i in range(X.shape[1]):
    #    print(X[:,i].dtype)
    feature_mean = np.mean(X, axis=0)
    #feature_var = np.var(X, axis=0)

    # Find variance for each class, maybe do normalization as well??
    # ID's for
    n_positive = (y == 1).sum()
    n_negative = (y == 0).sum()

    # Split positive and neg samples
    pos_samples = X[y == 1]
    neg_samples = X[y == 0]

    # get variance and mean for positive and negative labels for all features
    pos_variances = np.var(pos_samples, axis=0)
    neg_variances = np.var(neg_samples, axis=0)

    # get means
    pos_means = np.mean(pos_samples, axis=0)
    neg_means = np.mean(neg_samples, axis=0)

    #print(pos_variances)
    #print(neg_variances)

    # Calculate Fisher score for each feature
    Fr = np.zeros(X.shape[1])

    for i in range(X.shape[1]):
        Fr[i] = n_positive * np.power(pos_means[i] - feature_mean[i], 2) + \
                n_negative * np.power(neg_means[i] - feature_mean[i], 2)

        Fr[i] /= (n_positive * pos_variances[i] + n_negative * neg_variances[i])

    return Fr


class ModelEvaluation:
    def __init__(self, name, ):
        self.name = name
        self.accuracies = []
        self.precision = []
        self.recall = []
        self.f1 = []
        self.auc = []

    def add_to_file(self, filePath, text):
        with open(filePath, "a+") as f:
            f.write(text + "\n")

    def add_results(self, accuracy, precision, rc, f1, auc):
        self.accuracies.append(accuracy)
        self.precision.append(precision)
        self.recall.append(rc)
        self.f1.append(f1)
        self.auc.append(auc)

    def add_results_dict(self, results):
        self.accuracies.append(results["accuracy"])
        self.precision.append(results["precision"])
        self.recall.append(results["recall"])
        self.f1.append(results["f1"])
        self.auc.append(results["auc"])

    def print_statistics(self):
        print("Statistics for {}".format(self.name))
        print("Accuracy mean:", np.mean(self.accuracies))
        print("Accuracy std:", np.std(self.accuracies))
        print("Precision mean:", np.mean(self.precision))
        print("Precision std:", np.std(self.precision))
        print("Recall mean:", np.mean(self.recall))
        print("Recall std:", np.std(self.recall))
        print("F1 mean:", np.mean(self.f1))
        print("F1 std:", np.std(self.f1))
        print("AUC mean:", np.mean(self.auc))
        print("AUC std:", np.std(self.auc))
        print("")

    def print_statistics_drive(self):
        print("Statistics for {}".format(self.name))
        print("{} {} {} {} {} {} {} {} {} {}".format(np.mean(self.accuracies),
                                                     np.std(self.accuracies), np.mean(self.auc), np.std(self.auc),
                                                     np.mean(self.f1), np.std(self.f1),
                                                     np.mean(self.recall), np.std(self.recall), np.mean(self.precision),
                                                     np.std(self.precision)))
        print("")

    def write_statistics_file(self, filePath):
        text = "Statistics for {}".format(self.name) + "\n" + "{} {} {} {} {} {} {} {} {} {}".format(
            np.mean(self.accuracies),
            np.std(self.accuracies), np.mean(self.auc), np.std(self.auc), np.mean(self.f1), np.std(self.f1),
            np.mean(self.recall), np.std(self.recall), np.mean(self.precision), np.std(self.precision)) + "\n"

        self.add_to_file(filePath, text)

    def write_statistics_file_noname(self, filePath):
        text = "{} {} {} {} {} {} {} {} {} {}".format(
            np.mean(self.accuracies),
            np.std(self.accuracies), np.mean(self.auc), np.std(self.auc), np.mean(self.f1), np.std(self.f1),
            np.mean(self.recall), np.std(self.recall), np.mean(self.precision), np.std(self.precision))

        self.add_to_file(filePath, text)


class ExperimentRunner:

    def __init__(self, experiment_name, output_file, results_folder, inp_path, log_name, output_folder, log_template,
                 dt_max_depth=15, dt_min_leaf=10, selection_method="fisher", selection_counts=None,
                 coverage_threshold=None, sequence_threshold=5, payload=False, payload_settings=None,
                 reencode=False, payload_type=None, payload_dwd_settings=None):

        if not payload_type:
            self.payload_type = "normal"
        else:
            self.payload_type = payload_type


        self.payload_dwd_settings = payload_dwd_settings

        self.payload = payload
        self.payload_settings = payload_settings

        self.counter = 0

        self.reencode = reencode

        self.experiment_name = experiment_name
        self.output_file = output_file
        self.results_folder = results_folder
        self.dt_max_depth = dt_max_depth
        self.dt_min_leaf = dt_min_leaf

        self.inp_path = inp_path
        self.log_name = log_name
        self.output_folder = output_folder

        if not os.path.exists(output_folder):
            os.mkdir(output_folder)

        self.log_template = log_template
        self.log_path = self.output_folder + log_template
        self.log_path_seq = log_template

        self.train_output_file = "train_" + output_file
        self.test_output_file = "test_" + output_file

        if not coverage_threshold and (selection_method == "coverage" or selection_method == "rf_importance"):
            self.coverage_threshold = 20
        else:
            self.coverage_threshold = coverage_threshold

        self.sequence_threshold = sequence_threshold

        self.method = selection_method
        if not selection_counts and selection_method == "fisher":
            self.selection_counts = [100, 500, 1000]
        else:
            self.selection_counts = selection_counts
        self.encodings = ["tr", "tra", "mr", "mra"]



    def interpret_results(self, results, model_type, sequence_encoding=None):
        if self.method == "fisher":

            selection_models = defaultdict(dict)
            results_per_selection = defaultdict(list)

            for split in results:
                for train_test in split["result"]:
                    # Train results
                    train_results = train_test["train"]
                    test_results = train_test["test"]
                    selection_count = train_test["selection_count"]

                    res = {
                        "train": train_results,
                        "test": test_results
                    }

                    results_per_selection[selection_count].append(res)

            for selection_count in self.selection_counts:
                # for encoding per selection
                if model_type == "sequence":
                    test_model_eval = ModelEvaluation(
                        "TEST Model {} {} with selection {} with {} features".format(model_type, sequence_encoding,
                                                                                     self.method,
                                                                                     selection_count))
                    train_model_eval = ModelEvaluation(
                        "TRAIN Model {} {} with selection {} with {} features".format(model_type, sequence_encoding,
                                                                                      self.method,
                                                                                      selection_count))
                else:
                    test_model_eval = ModelEvaluation(
                        "TEST Model {} with selection {} with {} features".format(model_type,
                                                                                  self.method,
                                                                                  selection_count))
                    train_model_eval = ModelEvaluation(
                        "TRAIN Model {} with selection {} with {} features".format(model_type,
                                                                                   self.method,
                                                                                   selection_count))
                for result in results_per_selection[selection_count]:
                    train_model_eval.add_results_dict(result["train"])
                    test_model_eval.add_results_dict(result["test"])

                selection_models[selection_count]["train"] = train_model_eval
                selection_models[selection_count]["test"] = test_model_eval

            return selection_models

        else:
            models = defaultdict(dict)

            if model_type == "sequence":
                test_model_eval = ModelEvaluation(
                    "TEST Model {} {} with selection {}".format(model_type, sequence_encoding, self.method))
                train_model_eval = ModelEvaluation(
                    "TRAIN Model {} {} with selection {}".format(model_type, sequence_encoding, self.method))
            else:
                test_model_eval = ModelEvaluation(
                    "TEST Model {} with selection {}".format(model_type, self.method))
                train_model_eval = ModelEvaluation(
                    "TRAIN Model {} with selection {} ".format(model_type, self.method))

            for r in results:
                train_model_eval.add_results_dict(r["result"]["train"])
                test_model_eval.add_results_dict(r["result"]["test"])

            models["train"] = train_model_eval
            models["test"] = test_model_eval

            return models

    @staticmethod
    def generate_cross_validation_logs(log, log_name, output_folder):
        split_perc = 0.2
        log_size = len(log)
        partition_size = int(split_perc * log_size)
        for log_nr in range(5):
            new_log = XFactory.create_log(log.get_attributes().clone())
            for elem in log.get_extensions():
                new_log.get_extensions().add(elem)

            new_log.__classifiers = log.get_classifiers().copy()
            new_log.__globalTraceAttributes = log.get_global_trace_attributes().copy()
            new_log.__globalEventAttributes = log.get_global_event_attributes().copy()

            # Add first part.
            for i in range(0, (log_nr * partition_size)):
                new_log.append(log[i])

            # Add last part.
            for i in range((log_nr + 1) * partition_size, log_size):
                new_log.append(log[i])

            # This is the test partitions, added to end
            for i in range(log_nr * partition_size, (log_nr + 1) * partition_size):
                if i >= log_size:
                    break  # edge case
                new_log.append(log[i])

            with open(output_folder + "/" + log_name[:-4] + "_" + str(log_nr + 1) + ".xes", "w") as file:
                XesXmlSerializer().serialize(new_log, file)

            with open("logs/" + log_name[:-4] + "_" + str(log_nr + 1) + ".xes", "w") as file:
                XesXmlSerializer().serialize(new_log, file)

    @staticmethod
    def create_folder_structure(directory, payload=False, payload_type=None):
        if not os.path.exists(directory):
            os.makedirs(directory)

            # first level
            for i in range(1, 6):
                os.makedirs(directory + "/" + "split" + str(i))

                # second level
                os.makedirs(directory + "/" + "split" + str(i) + "/" + "base")
                os.makedirs(directory + "/" + "split" + str(i) + "/" + "declare")
                os.makedirs(directory + "/" + "split" + str(i) + "/" + "mr")
                os.makedirs(directory + "/" + "split" + str(i) + "/" + "mra")
                os.makedirs(directory + "/" + "split" + str(i) + "/" + "tr")
                os.makedirs(directory + "/" + "split" + str(i) + "/" + "tra")

                if payload:
                    if payload_type == "normal" or "both":
                        os.makedirs(directory + "/" + "split" + str(i) + "/" + "payload")
                    if payload_type == "dwd" or "both":
                        os.makedirs(directory + "/" + "split" + str(i) + "/" + "dwd")

    @staticmethod
    def cross_validation_pipeline(inp_path, log_name, output_folder):
        # 1. Load file
        log = read_XES_log(inp_path + "/" + log_name)

        # 2. Randomize order of traces.
        shuffle(log)

        # 3. Split into 5 parts for cross validation
        ExperimentRunner.generate_cross_validation_logs(log, log_name, output_folder)

    @staticmethod
    def read_baseline_log(results_folder, split_nr):
        split = "split" + str(split_nr)
        encoding = "base"

        file_loc = results_folder + "/" + split + "/" + encoding
        train_path = file_loc + "/" + "baseline_train.csv"
        test_path = file_loc + "/" + "baseline_test.csv"
        train_df = pd.read_csv(train_path, sep=",", index_col="Case_ID", na_filter=False)
        test_df = pd.read_csv(test_path, sep=",", index_col="Case_ID", na_filter=False)

        return train_df, test_df

    @staticmethod
    def read_payload_log(results_folder, split_nr):
        split = "split" + str(split_nr)
        encoding = "payload"

        file_loc = results_folder + "/" + split + "/" + encoding
        train_path = file_loc + "/" + "payload_train.csv"
        test_path = file_loc + "/" + "payload_test.csv"
        train_df = pd.read_csv(train_path, sep=",", index_col="Case_ID", na_filter=False)
        test_df = pd.read_csv(test_path, sep=",", index_col="Case_ID", na_filter=False)

        return train_df, test_df

    @staticmethod
    def read_declare_with_data_log(results_folder, split_nr):
        split = "split" + str(split_nr)
        encoding = "dwd"

        file_loc = results_folder + "/" + split + "/" + encoding
        train_path = file_loc + "/" + "dwd_train.csv"
        test_path = file_loc + "/" + "dwd_test.csv"
        train_df = pd.read_csv(train_path, sep=",", index_col="Case_ID", na_filter=False)
        test_df = pd.read_csv(test_path, sep=",", index_col="Case_ID", na_filter=False)

        return train_df, test_df


    @staticmethod
    def read_declare_log(results_folder, split_nr):
        split = "split" + str(split_nr)
        encoding = "declare"

        file_loc = results_folder + "/" + split + "/" + encoding
        train_path = file_loc + "/" + "declare_train.csv"
        test_path = file_loc + "/" + "declare_test.csv"
        train_df = pd.read_csv(train_path, sep=",", index_col="Case_ID", na_filter=False)
        test_df = pd.read_csv(test_path, sep=",", index_col="Case_ID", na_filter=False)

        return train_df, test_df

    @staticmethod
    def read_sequence_log(results_folder, encoding, split_nr):
        split = "split" + str(split_nr)
        file_loc = results_folder + "/" + split + "/" + encoding
        train_path = file_loc + "/" + "globalLog.csv"
        global_df = pd.read_csv(train_path, sep=";", index_col="Case_ID", na_filter=False)

        size_df = len(global_df)

        train_size = int(0.8 * size_df)
        train_df = global_df.iloc[:train_size, ]
        test_df = global_df.iloc[train_size:, ]

        return train_df, test_df

    @staticmethod
    def evaluate_model(clf, X_train, y_train, X_test, y_test) -> (dict, dict):
        """
        Evaluates the model
        :param y_test:
        :param predictions:
        :param probabilities:
        :return:
        """

        # predict on train data

        predictions = clf.predict(X_train)
        probabilities = clf.predict_proba(X_train).T[1]


        # get metrics
        train_accuracy = accuracy_score(y_train, predictions)
        train_precision = precision_score(y_train, predictions)
        train_rc = recall_score(y_train, predictions)
        train_f1 = f1_score(y_train, predictions)
        train_auc = roc_auc_score(y_train, probabilities)

        # predict on test data
        predictions = clf.predict(X_test)
        probabilities = clf.predict_proba(X_test).T[1]

        # get metrics
        test_accuracy = accuracy_score(y_test, predictions)
        test_precision = precision_score(y_test, predictions)
        test_rc = recall_score(y_test, predictions)
        test_f1 = f1_score(y_test, predictions)
        test_auc = roc_auc_score(y_test, probabilities)

        train_results = {
            "accuracy": train_accuracy,
            "precision": train_precision,
            "recall": train_rc,
            "f1": train_f1,
            "auc": train_auc
        }

        test_results = {
            "accuracy": test_accuracy,
            "precision": test_precision,
            "recall": test_rc,
            "f1": test_f1,
            "auc": test_auc
        }

        return train_results, test_results

    def feature_selection(self, train_df, test_df, y_train, params, payload_train_df=None, payload_test_df=None, ):


        if payload_train_df is not None:
            train_df = pd.concat([train_df, payload_train_df], axis=1)
            test_df = pd.concat([test_df, payload_test_df], axis=1)


        #if payload_train_df is not None:
        #    X_train_df = pd.concat([train_df.iloc[:, selected_ranks[:chosen]], payload_train_df], axis=1)
        #    X_test_df = pd.concat([test_df.iloc[:, selected_ranks[:chosen]], payload_test_df], axis=1)
        #    feature_names = X_train_df.columns
        #    X_train = X_train_df.values
        #    X_test = X_test_df.values
        train_df = train_df.transpose().drop_duplicates().transpose()
        remaining_columns = train_df.columns

        test_df = test_df[remaining_columns]

        # remove no-variance, constants
        train_df = train_df.loc[:, (train_df != train_df.iloc[0]).any()]
        test_df = test_df[train_df.columns]

        # Turn into np object
        X_train = train_df.values
        X_test = test_df.values


        # Same from payload data if needed
        #if payload_train_df is not None:
        #    payload_train_df = payload_train_df.loc[:, (payload_train_df != payload_train_df.iloc[0]).any()]
        #    payload_test_df = payload_test_df[payload_train_df.columns]

            # remove duplicates
         #   payload_train_df = payload_train_df.transpose().drop_duplicates().transpose()
         #   payload_remaining_columns = payload_train_df.columns
         #   payload_test_df = payload_test_df[payload_remaining_columns]




        # No feature selection performed on payload..

        #sel = VarianceThreshold()
        #sel.fit(X_train)
        #X_train = sel.transform(X_train)
        #X_test = sel.transform(X_test)
        feature_names = None

        selection_method = self.method

        if selection_method == "PCA":
            # PCA
            standardizer = StandardScaler().fit(X_train)

            # Standardize first
            X_train = standardizer.transform(X_train)
            X_test = standardizer.transform(X_test)

            # Apply PCA
            pca = PCA(n_components=3)

            pca.fit(X_train)
            X_train = pca.transform(X_train)
            X_test = pca.transform(X_test)

            feature_names = None

        elif selection_method == "chi2":

            sel_count = self.selection_counts

            fitt = SelectKBest(chi2, k=min(X_train.shape[1], sel_count)).fit(X_train, y_train)

            X_train = fitt.transform(X_train)
            X_test = fitt.transform(X_test)

            feature_names = train_df.columns[fitt.get_support()]


        elif selection_method == "fisher":
            sel_count = params["selection_count"]
            scores = fisher_calculation(X_train, y_train)
            #scores = fisher_score.fisher_score(X_train, y_train)
            selected_ranks = fisher_score.feature_ranking(scores)[:sel_count]

            X_train = X_train[:, selected_ranks]
            X_test = X_test[:, selected_ranks]
            for i, rank in enumerate(selected_ranks[:10]):
                print(train_df.columns[rank], scores[i])

            feature_names = train_df.columns[selected_ranks]

        elif selection_method == "coverage":
            """
            scores = fisher_calculation(X_train, y_train)
            selected_ranks = fisher_score.feature_ranking(scores)
            threshold = self.coverage_threshold

            # Start selecting from selected_ranks until every trace is covered N times
            trace_remaining = dict()
            for i, trace_name in enumerate(train_df.index.values):
                trace_remaining[i] = threshold

            chosen = 0
            chosen_ranks = []
            # Go from higher to lower
            for rank in selected_ranks:
                is_chosen = False
                if len(trace_remaining) == 0:
                    break
                chosen += 1
                # Get column
                marked_for_deletion = set()
                for k in trace_remaining.keys():
                    if train_df.iloc[k, rank] > 0:
                        if not is_chosen:
                            # Only choose as a feature, if there is at least one trace covered by it.
                            chosen_ranks.append(rank)
                            is_chosen = True

                        trace_remaining[k] -= 1
                        if trace_remaining[k] <= 0:
                            marked_for_deletion.add(k)

                for k in marked_for_deletion:
                    del trace_remaining[k]

            X_train = X_train[:, selected_ranks[chosen_ranks]]
            X_test = X_test[:, selected_ranks[chosen_ranks]]
            feature_names = train_df.columns[selected_ranks[chosen_ranks]]
            """

            # Alternative version
            scores = fisher_calculation(X_train, y_train)
            selected_ranks = fisher_score.feature_ranking(scores)
    
            threshold = self.coverage_threshold
    
            # Start selecting from selected_ranks until every trace is covered N times
            trace_remaining = dict()
            for i, trace_name in enumerate(train_df.index.values):
                trace_remaining[i] = threshold
    
            chosen = 0
            #chosen_ranks = []
            # Go from higher to lower
            for rank in selected_ranks:
                #is_chosen = False
                if len(trace_remaining) == 0:
                    break
                chosen += 1
                # Get column
                marked_for_deletion = set()
                for k in trace_remaining.keys():
                    if train_df.iloc[k, rank] > 0:
                        #if not is_chosen:
                            # Only choose as a feature, if there is at least one trace covered by it.
                            #chosen_ranks.append(rank)
                            #is_chosen = True
    
                        trace_remaining[k] -= 1
                        if trace_remaining[k] <= 0:
                            marked_for_deletion.add(k)
    
                for k in marked_for_deletion:
                    del trace_remaining[k]
    
            X_train = X_train[:, selected_ranks[:chosen]]
            X_test = X_test[:, selected_ranks[:chosen]]
    
            feature_names = train_df.columns[selected_ranks[:chosen]]


        #if payload_train_df is not None:
        #    X_train_df = pd.concat([train_df.iloc[:, selected_ranks[:chosen]], payload_train_df], axis=1)
        #    X_test_df = pd.concat([test_df.iloc[:, selected_ranks[:chosen]], payload_test_df], axis=1)
        #    feature_names = X_train_df.columns
        #    X_train = X_train_df.values
        #    X_test = X_test_df.values
        #else:

        #print(feature_names)


        return X_train, X_test, feature_names

    def train_and_evaluate_select(self, train_df: pd.DataFrame, test_df: pd.DataFrame,
                                  payload_train_df=None, payload_test_df=None, params=None,
                                  exp_name=None, split_nr=None) -> (dict, dict):
        """
        Trains and evaluates model
        :param train_df:
        :param test_df:
        :param params:
        :return:
        """

        self.counter += 1

        y_train = train_df.pop('Label').values
        y_test = test_df.pop('Label').values
        X_train, X_test, feature_names = self.feature_selection(train_df, test_df, y_train, params=params,
                                                                payload_train_df=payload_train_df,
                                                                payload_test_df=payload_test_df)

        # Save input to arff file to be used for RIPPER!
        SAVE_ARFF = False

        if SAVE_ARFF:
            encoded_features = [str(i) for i in range(len(feature_names)+1)]
            features = list(feature_names)
            features.append("Label")

            with open("arff/feature_encoding_{}".format(self.counter), "w") as fc:
                feature_pairs = zip(encoded_features, features)
                for k, v in feature_pairs:
                    fc.write(k + ":" + v + "\n")

            train_new = pd.DataFrame(X_train, columns=feature_names)
            test_new = pd.DataFrame(X_test, columns=feature_names)
            train_new["Label"] = y_train
            test_new["Label"] = y_test
            train_new["Label"] = train_new["Label"].astype("category")
            test_new["Label"] = test_new["Label"].astype('category')

            arff.dump('arff/train_data_{}.arff'.format(self.counter)
                      , train_new.values
                      , relation='data_arff'
                      , names=encoded_features)
            arff.dump('arff/test_data_{}.arff'.format(self.counter)
                      , test_new.values
                      , relation='data_arff'
                      , names=encoded_features)

        # Toggle this to save snapshots
        SAVE_CSV = False
        if SAVE_CSV and exp_name and split_nr:
            ## Save all
            new_feature_names = list(map(lambda x: x.replace(",", "."), feature_names))
            new_feature_names = list(map(lambda x: x.replace('"', ""), new_feature_names))
            new_feature_names = list(map(lambda x: x.replace("'", ""), new_feature_names))
            #new_feature_names = [str(i) for i in range(len(new_feature_names))]

            savename ="synthmra_{}_{}_{}.csv".format(exp_name, self.coverage_threshold, split_nr)
            train_new = pd.DataFrame(X_train, columns=new_feature_names)
            test_new = pd.DataFrame(X_test, columns= new_feature_names)
            train_new["Label"] = y_train
            test_new["Label"] = y_test


            # Save separately for each split, merged encoding type. To feed into RIPPER
            train_new.to_csv("snapshots/train_" + savename, index=False)
            test_new.to_csv("snapshots/test_" + savename, index=False)


        # Train classifier
        clf = DecisionTreeClassifier(max_depth=self.dt_max_depth, min_samples_leaf=self.dt_min_leaf)
        clf.fit(X_train, y_train)

        # True to export tree .dot file
        export_tree = False
        if export_tree:
            export_graphviz(clf, out_file="outputfile_{}.dot".format(self.counter), feature_names=feature_names)

        #tree_to_code(clf, feature_names)

        print_importances = False
        if print_importances:
            print(list(reversed(sorted(zip(feature_names, clf.feature_importances_), key=lambda x: x[1])))[:10])

        # Evaluate model
        train_results, test_results = ExperimentRunner.evaluate_model(clf, X_train, y_train, X_test, y_test)

        return train_results, test_results

    def train(self, train_df, test_df, payload_train_df=None, payload_test_df=None, split_nr=None, exp_name=None):
        if self.method == "fisher":
            selection_counts = self.selection_counts
            results = []
            # Trying all selection counts
            for selection_count in selection_counts:
                if payload_train_df is not None:
                    train_results, test_results = self.train_and_evaluate_select(train_df.copy(), test_df.copy(),
                                                                             payload_train_df.copy(), payload_test_df.copy(),                                                                         params={"selection_count": selection_count})
                else:
                    train_results, test_results = self.train_and_evaluate_select(train_df.copy(), test_df.copy(),
                                                                                 params={"selection_count": selection_count})
                result = {
                    "train": train_results,
                    "test": test_results,
                    "selection_count": selection_count
                }

                results.append(result)

            return results
        else:
            if payload_train_df is not None:
                train_results, test_results = self.train_and_evaluate_select(train_df.copy(),
                                                                             test_df.copy(),
                                                                             payload_train_df.copy(),
                                                                             payload_test_df.copy(),
                                                                             split_nr=split_nr, exp_name=exp_name)
            else:
                train_results, test_results = self.train_and_evaluate_select(train_df.copy(),
                                                                            test_df.copy(),
                                                                            split_nr=split_nr, exp_name=exp_name)

            result = {
                "train": train_results,
                "test": test_results
            }

            return result

    def baseline_train(self):
        """

        :return:
        """

        results = []
        for split_nr in range(1, 6):
            train_df, test_df = ExperimentRunner.read_baseline_log(self.results_folder, split_nr)

            tr_result = self.train(train_df, test_df, split_nr=split_nr, exp_name="baseline")

            result = {
                "result": tr_result,
                "split": split_nr
            }

            results.append(result)

        return results

    def declare_train(self):
        """
        Train and evaluate declare models
        :return:
        """

        results = []

        # Separately for every split. Reduce total number of file parsing.
        for split_nr in range(1, 6):
            train_df, test_df = ExperimentRunner.read_declare_log(self.results_folder, split_nr)
            tr_result = self.train(train_df, test_df, split_nr=split_nr, exp_name="declare")

            result = {
                "result": tr_result,
                "split": split_nr
            }

            results.append(result)

        return results

    def sequence_train(self, encoding):
        """
        Trains a sequence model with given encoding
        :param encoding: sequence encoding
        :return:
        """

        results = []
        for split_nr in range(1, 6):
            # Read the log
            train_df, test_df = ExperimentRunner.read_sequence_log(self.results_folder, encoding, split_nr)

            tr_result = self.train(train_df, test_df, split_nr=split_nr, exp_name="sequence_{}".format(encoding))

            result = {
                "result": tr_result,
                "split": split_nr,
                "encoding": encoding
            }

            results.append(result)

        return results

    def hybrid_train(self):
        """
        Hybrid model training
        :return:
        """
        encodings = ["mr", "mra", "tr", "tra"]

        results = []
        for split_nr in range(1, 6):
            dec_train_df, dec_test_df = ExperimentRunner.read_declare_log(self.results_folder, split_nr)
            seq_train_list = []
            seq_test_list = []
            for encoding in encodings:
                seq_train_df, seq_test_df = ExperimentRunner.read_sequence_log(self.results_folder, encoding, split_nr)

                seq_train_df = seq_train_df.drop("Label", axis=1)
                seq_test_df = seq_test_df.drop("Label", axis=1)

                new_columns = [column + "_{}".format(encoding) for column in seq_train_df.columns]
                seq_train_df.columns = new_columns
                seq_test_df.columns = new_columns

                seq_train_list.append(seq_train_df)
                seq_test_list.append(seq_test_df)

            merged_train_df = pd.concat([dec_train_df] + seq_train_list, axis=1)
            merged_test_df = pd.concat([dec_test_df] + seq_test_list, axis=1)

            tr_result = self.train(merged_train_df, merged_test_df, split_nr=split_nr, exp_name="hybrid")
            result = {
                "result": tr_result,
                "split": split_nr
            }

            results.append(result)

        return results

    def payload_train(self):
        """
        Trains and tests models just on payload data.
        :return:
        """
        results = []
        for split_nr in range(1, 6):
            baseline_train_df, baseline_test_df = ExperimentRunner.read_baseline_log(self.results_folder, split_nr)
            payload_train_df, payload_test_df = ExperimentRunner.read_payload_log(self.results_folder, split_nr)

            payload_train_df["Label"] = baseline_train_df["Label"]
            payload_test_df["Label"] = baseline_test_df["Label"]

            tr_result = self.train(payload_train_df, payload_test_df, split_nr=split_nr, exp_name="payload")

            result = {
                "result": tr_result,
                "split": split_nr
            }

            results.append(result)

        return results

    def baseline_train_with_data(self):
        """

        :return:
        """

        results = []
        for split_nr in range(1, 6):
            train_df, test_df = ExperimentRunner.read_baseline_log(self.results_folder, split_nr)
            payload_train_df, payload_test_df = ExperimentRunner.read_payload_log(self.results_folder, split_nr)

            #merged_train_df = pd.concat([train_df, payload_train_df], axis=1)
            #merged_test_df = pd.concat([test_df, payload_test_df], axis=1)

            tr_result = self.train(train_df, test_df, payload_train_df, payload_test_df,
                                   split_nr=split_nr, exp_name="baseline_payload")


            result = {
                "result": tr_result,
                "split": split_nr
            }

            results.append(result)

        return results


    def baseline_train_with_dwd(self):
        """

        :return:
        """

        results = []
        for split_nr in range(1, 6):
            train_df, test_df = ExperimentRunner.read_baseline_log(self.results_folder, split_nr)
            payload_train_df, payload_test_df = ExperimentRunner.read_declare_with_data_log(self.results_folder, split_nr)

            # merged_train_df = pd.concat([train_df, payload_train_df], axis=1)
            # merged_test_df = pd.concat([test_df, payload_test_df], axis=1)

            tr_result = self.train(train_df, test_df, payload_train_df, payload_test_df,
                                   split_nr=split_nr, exp_name="baseline_dwd")

            result = {
                "result": tr_result,
                "split": split_nr
            }

            results.append(result)

        return results


    def declare_train_with_data(self):
        """
        Train and evaluate declare models
        :return:
        """

        results = []
        # Separately for every split. Reduce total number of file parsing.
        for split_nr in range(1, 6):
            train_df, test_df = ExperimentRunner.read_declare_log(self.results_folder, split_nr)
            payload_train_df, payload_test_df = ExperimentRunner.read_payload_log(self.results_folder, split_nr)


            tr_result = self.train(train_df, test_df, payload_train_df, payload_test_df,
                                   split_nr=split_nr, exp_name="declare_data")


            result = {
                "result": tr_result,
                "split": split_nr
            }

            results.append(result)

        return results


    def declare_train_with_dwd(self):
        """
        Train and evaluate declare models
        :return:
        """

        results = []
        # Separately for every split. Reduce total number of file parsing.
        for split_nr in range(1, 6):
            train_df, test_df = ExperimentRunner.read_declare_log(self.results_folder, split_nr)
            payload_train_df, payload_test_df = ExperimentRunner.read_declare_with_data_log(self.results_folder, split_nr)

            tr_result = self.train(train_df, test_df, payload_train_df, payload_test_df,
                                   split_nr=split_nr, exp_name="declare_dwd")


            result = {
                "result": tr_result,
                "split": split_nr
            }

            results.append(result)

        return results


    def declare_train_with_dwd_data(self):
        """
        Train and evaluate declare models
        :return:
        """
        results = []
        # Separately for every split. Reduce total number of file parsing.
        for split_nr in range(1, 6):
            train_df, test_df = ExperimentRunner.read_declare_log(self.results_folder, split_nr)
            payload_train_df, payload_test_df = ExperimentRunner.read_payload_log(self.results_folder, split_nr)
            payload_train_df_2, payload_test_df_2 = ExperimentRunner.read_declare_with_data_log(self.results_folder, split_nr)

            merged_train_df = pd.concat([train_df, payload_train_df_2], axis=1)
            merged_test_df = pd.concat([test_df, payload_test_df_2], axis=1)

            tr_result = self.train(merged_train_df, merged_test_df, payload_train_df, payload_test_df,
                                   split_nr=split_nr, exp_name="declare_dwd_data")


            result = {
                "result": tr_result,
                "split": split_nr
            }

            results.append(result)

        return results


    def sequence_train_with_data(self, encoding):
        """
        Trains a sequence model with given encoding
        :param encoding: sequence encoding
        :return:
        """

        results = []
        for split_nr in range(1, 6):
            # Read the log
            train_df, test_df = ExperimentRunner.read_sequence_log(self.results_folder, encoding, split_nr)
            payload_train_df, payload_test_df = ExperimentRunner.read_payload_log(self.results_folder, split_nr)

            #merged_train_df = pd.concat([train_df, payload_train_df], axis=1)
            #merged_test_df = pd.concat([test_df, payload_test_df], axis=1)

            tr_result = self.train(train_df, test_df, payload_train_df, payload_test_df,
                                   split_nr=split_nr, exp_name="sequence_data_{}".format(encoding))


            result = {
                "result": tr_result,
                "split": split_nr,
                "encoding": encoding
            }

            results.append(result)

        return results

    def sequence_train_with_dwd(self, encoding):
        """
        Trains a sequence model with given encoding
        :param encoding: sequence encoding
        :return:
        """

        results = []
        for split_nr in range(1, 6):
            # Read the log
            train_df, test_df = ExperimentRunner.read_sequence_log(self.results_folder, encoding, split_nr)
            payload_train_df, payload_test_df = ExperimentRunner.read_declare_with_data_log(self.results_folder, split_nr)

            #merged_train_df = pd.concat([train_df, payload_train_df], axis=1)
            #merged_test_df = pd.concat([test_df, payload_test_df], axis=1)

            tr_result = self.train(train_df, test_df, payload_train_df, payload_test_df, split_nr=split_nr, exp_name="sequence_dwd")


            result = {
                "result": tr_result,
                "split": split_nr,
                "encoding": encoding
            }

            results.append(result)

        return results

    def hybrid_with_data(self):
        """
        Hybrid model training with additional data
        :return:
        """

        encodings = ["mr", "mra", "tr", "tra"]

        results = []
        for split_nr in range(1, 6):
            dec_train_df, dec_test_df = ExperimentRunner.read_declare_log(self.results_folder, split_nr)
            payload_train_df, payload_test_df = ExperimentRunner.read_payload_log(self.results_folder, split_nr)
            seq_train_list = []
            seq_test_list = []
            for encoding in encodings:
                seq_train_df, seq_test_df = ExperimentRunner.read_sequence_log(self.results_folder, encoding, split_nr)

                seq_train_df = seq_train_df.drop("Label", axis=1)
                seq_test_df = seq_test_df.drop("Label", axis=1)

                new_columns = [column + "_{}".format(encoding) for column in seq_train_df.columns]
                seq_train_df.columns = new_columns
                seq_test_df.columns = new_columns

                seq_train_list.append(seq_train_df)
                seq_test_list.append(seq_test_df)

            merged_train_df = pd.concat([dec_train_df] + seq_train_list, axis=1)
            merged_test_df = pd.concat([dec_test_df] + seq_test_list, axis=1)
            # , payload_train_df, payload_test_df
            tr_result = self.train(merged_train_df, merged_test_df, payload_train_df, payload_test_df, split_nr=split_nr, exp_name="hybrid_data")
            result = {
                "result": tr_result,
                "split": split_nr
            }

            results.append(result)

        return results


    def hybrid_with_dwd(self):
        """
        Hybrid model training with additional data
        :return:
        """

        encodings = ["mr", "mra", "tr", "tra"]

        results = []
        for split_nr in range(1, 6):
            dec_train_df, dec_test_df = ExperimentRunner.read_declare_log(self.results_folder, split_nr)
            payload_train_df, payload_test_df = ExperimentRunner.read_declare_with_data_log(self.results_folder, split_nr)
            seq_train_list = []
            seq_test_list = []
            for encoding in encodings:
                seq_train_df, seq_test_df = ExperimentRunner.read_sequence_log(self.results_folder, encoding, split_nr)

                seq_train_df = seq_train_df.drop("Label", axis=1)
                seq_test_df = seq_test_df.drop("Label", axis=1)

                new_columns = [column + "_{}".format(encoding) for column in seq_train_df.columns]
                seq_train_df.columns = new_columns
                seq_test_df.columns = new_columns

                seq_train_list.append(seq_train_df)
                seq_test_list.append(seq_test_df)

            merged_train_df = pd.concat([dec_train_df] + seq_train_list, axis=1)
            merged_test_df = pd.concat([dec_test_df] + seq_test_list, axis=1)
            # , payload_train_df, payload_test_df
            tr_result = self.train(merged_train_df, merged_test_df, payload_train_df, payload_test_df, split_nr=split_nr, exp_name="hybrid_dwd")
            result = {
                "result": tr_result,
                "split": split_nr
            }

            results.append(result)

        return results

    def hybrid_with_dwd_and_payload(self):
        """
        Hybrid model training with additional data
        :return:
        """

        encodings = ["mr", "mra", "tr", "tra"]

        results = []
        for split_nr in range(1, 6):
            dec_train_df, dec_test_df = ExperimentRunner.read_declare_log(self.results_folder, split_nr)
            payload_train_df, payload_test_df = ExperimentRunner.read_declare_with_data_log(self.results_folder, split_nr)
            payload_train_df_2, payload_test_df_2 = ExperimentRunner.read_payload_log(self.results_folder, split_nr)
            seq_train_list = []
            seq_test_list = []
            for encoding in encodings:
                seq_train_df, seq_test_df = ExperimentRunner.read_sequence_log(self.results_folder, encoding, split_nr)

                seq_train_df = seq_train_df.drop("Label", axis=1)
                seq_test_df = seq_test_df.drop("Label", axis=1)

                new_columns = [column + "_{}".format(encoding) for column in seq_train_df.columns]
                seq_train_df.columns = new_columns
                seq_test_df.columns = new_columns

                seq_train_list.append(seq_train_df)
                seq_test_list.append(seq_test_df)

            merged_train_df = pd.concat([dec_train_df, payload_train_df_2] + seq_train_list, axis=1)
            merged_test_df = pd.concat([dec_test_df, payload_test_df_2] + seq_test_list, axis=1)
            # , payload_train_df, payload_test_df
            tr_result = self.train(merged_train_df, merged_test_df, payload_train_df, payload_test_df, split_nr=split_nr, exp_name="hybrid_dwd_payload")
            result = {
                "result": tr_result,
                "split": split_nr
            }

            results.append(result)

        return results


    def train_and_eval_benchmark(self):

        all_results = {}
        # MAKE SURE ALL METHODS USED ARE HERE. AND METHODS NOT USED ARE NOT!
        if not self.payload:
            print_order = ["bs", "dc", "tr", "tra", "mr", "mra", "hybrid"]
            print("Started working on baseline.")
            baseline_results = self.baseline_train()
            all_results["bs"] = self.interpret_results(baseline_results, "baseline")

            print("Started working on declare.")
            declare_results = self.declare_train()
            all_results["dc"] = self.interpret_results(declare_results, "declare")

            print("Started working on sequenceMR.")
            sequence_results = self.sequence_train("mr")
            all_results["mr"] = self.interpret_results(sequence_results, "sequence", "mr")

            print("Started working on sequenceTR.")
            sequence_results = self.sequence_train("tr")
            all_results["tr"] = self.interpret_results(sequence_results, "sequence", "tr")

            print("Started working on sequenceTRA.")
            sequence_results = self.sequence_train("tra")
            all_results["tra"] = self.interpret_results(sequence_results, "sequence", "tra")

            print("Started working on sequenceMRA.")
            sequence_results = self.sequence_train("mra")
            all_results["mra"] = self.interpret_results(sequence_results, "sequence", "mra")

            print("Started working on hybrid.")
            hybrid_results = self.hybrid_train()
            all_results["hybrid"] = self.interpret_results(hybrid_results, "hybrid")

        if self.payload:
            print_order = []
            if self.payload_type == "normal":
                print_order += ["payload", "bs_data", "dc_data", "tr_data", "tra_data", "mr_data", "mra_data", "hybrid_data"]
                print("Started working on payload train.")
                payload_results = self.payload_train()
                all_results["payload"] = self.interpret_results(payload_results, "payload")

                print("Started working on baseline with data.")
                baseline_results = self.baseline_train_with_data()
                all_results["bs_data"] = self.interpret_results(baseline_results, "baseline")

                print("Started working on declare with data.")
                declare_results = self.declare_train_with_data()
                all_results["dc_data"] = self.interpret_results(declare_results, "declare")

                print("Started working on sequenceMR with data.")
                sequence_results = self.sequence_train_with_data("mr")
                all_results["mr_data"] = self.interpret_results(sequence_results, "sequence", "mr")

                print("Started working on sequenceTR with data.")
                sequence_results = self.sequence_train_with_data("tr")
                all_results["tr_data"] = self.interpret_results(sequence_results, "sequence", "tr")

                print("Started working on sequenceTRA with data.")
                sequence_results = self.sequence_train_with_data("tra")
                all_results["tra_data"] = self.interpret_results(sequence_results, "sequence", "tra")

                print("Started working on sequenceMRA with data.")
                sequence_results = self.sequence_train_with_data("mra")
                all_results["mra_data"] = self.interpret_results(sequence_results, "sequence", "mra")

                print("Started working on hybrid with data.")
                payload_results = self.hybrid_with_data()
                all_results["hybrid_data"] = self.interpret_results(payload_results, "hybrid_data")

            if self.payload_type == "both":
                print_order += ["bs", "dc", "dc_data", "dc_dwd",  "dc_dwd_payload", "hybrid", "hybrid_data", "hybrid_dwd", "hybrid_dwd_payload"]

                print("Started working on baseline.")
                baseline_results = self.baseline_train()
                all_results["bs"] = self.interpret_results(baseline_results, "baseline")

                print("Started working on declare.")
                declare_results = self.declare_train()
                all_results["dc"] = self.interpret_results(declare_results, "declare")

                print("Started working on declare with payload.")
                declare_results = self.declare_train_with_data()
                all_results["dc_data"] = self.interpret_results(declare_results, "declare_payload")

                print("Started working on declare with dwd.")
                declare_results = self.declare_train_with_dwd()
                all_results["dc_dwd"] = self.interpret_results(declare_results, "declare_dwd")

                print("Started working on declare with dwd and payload.")
                declare_results = self.declare_train_with_dwd_data()
                all_results["dc_dwd_payload"] = self.interpret_results(declare_results, "declare_payload_dwd")


                print("Started working on hybrid.")
                payload_results = self.hybrid_train()
                all_results["hybrid"] = self.interpret_results(payload_results, "hybrid")

                print("Started working on hybrid with data.")
                payload_results = self.hybrid_with_data()
                all_results["hybrid_data"] = self.interpret_results(payload_results, "hybrid_data")

                print("Started working on hybrid with dwd.")
                payload_results = self.hybrid_with_dwd()
                all_results["hybrid_dwd"] = self.interpret_results(payload_results, "hybrid_dwd")

                print("Started working on hybrid with dwd and usual payload.")
                payload_results = self.hybrid_with_dwd_and_payload()
                all_results["hybrid_dwd_payload"] = self.interpret_results(payload_results, "hybrid_data_dwd")



        if self.method == "fisher":
            for selection_count in self.selection_counts:
                # find the print
                for meth in print_order:
                    for k, v in all_results.items():
                        if meth == k:
                            print(k)
                            evalTrain = v[selection_count]["train"]
                            evalTest = v[selection_count]["test"]
                            evalTrain.print_statistics_drive()
                            evalTest.print_statistics_drive()
                            evalTrain.write_statistics_file_noname(self.train_output_file)
                            evalTest.write_statistics_file_noname(self.test_output_file)
        else:
            for meth in print_order:
                for k, v in all_results.items():
                    if meth == k:
                        print(k)

                        evalTrain = v["train"]
                        evalTest = v["test"]
                        evalTrain.print_statistics_drive()
                        evalTest.print_statistics_drive()
                        evalTrain.write_statistics_file_noname(self.train_output_file)
                        evalTest.write_statistics_file_noname(self.test_output_file)

    def prepare_cross_validation(self):
        self.cross_validation_pipeline(self.inp_path, self.log_name, self.output_folder)

    def prepare_data(self):
        self.create_folder_structure(self.results_folder, payload=self.payload, payload_type=self.payload_type)
        run_baseline(self.experiment_name, self.log_path, self.results_folder)
        run_deviance_new(self.log_path, self.results_folder, reencode=self.reencode)
        run_sequences(self.log_path_seq, self.results_folder, sequence_threshold=self.sequence_threshold)

        if self.payload:
            if self.payload_type == "normal" or self.payload_type == "both":
                run_payload_extractor(self.log_path, self.payload_settings, self.results_folder)

            if self.payload_type == "dwd" or self.payload_type == "both":
                run_declare_with_data(self.log_path, self.payload_dwd_settings, self.results_folder)

    def clean_data(self):
        shutil.rmtree(self.results_folder)

