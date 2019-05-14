"""
Last version for data-aware declare mining
"""

from declaretemplates_data import *
from deviancecommon import read_XES_log, xes_to_data_positional
from declaredevmining import split_log_train_test, extract_unique_events_transformed
from declaredevmining import filter_candidates_by_support, count_classes

from skfeature.function.similarity_based import fisher_score
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.tree import DecisionTreeClassifier, _tree, export_graphviz




import numpy as np
import pandas as pd

import os
import shutil


def fisher_calculation(X, y):
    """
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


def apply_template_to_log(template, candidate, log):
    results = []
    for trace in log:
        result, vacuity = apply_template(template, trace, candidate)

        results.append(result)

    return results


def apply_data_template_to_log(template, candidate, log):
    results = []
    for trace in log:
        result, vacuity, fulfillments, violations = apply_data_template(template, trace, candidate)
        results.append((result, fulfillments, violations))

    return results


def generate_train_candidate_constraints(candidates, templates, train_log, constraint_support_norm,
                                         constraint_support_dev, filter_t=True):
    all_results = {}
    for template in templates:
        print("Started working on {}".format(template))
        for candidate in candidates:
            if len(candidate) == template_sizes[template]:
                constraint_result = apply_template_to_log(template, candidate, train_log)
                satis_normal, satis_deviant = find_if_satisfied_by_class(constraint_result, train_log,
                                                                         constraint_support_norm,
                                                                         constraint_support_dev)

                if not filter_t or (satis_normal or satis_deviant):
                    all_results[(template, candidate)] = constraint_result


    return all_results


def generate_test_candidate_constraints(candidates, templates, test_log, train_results):
    all_results = {}
    for template in templates:
        print("Started working on {}".format(template))
        for candidate in candidates:
            if len(candidate) == template_sizes[template]:
                if (template, candidate) in train_results:
                    constraint_result = apply_template_to_log(template, candidate, test_log)

                    all_results[(template, candidate)] = constraint_result

    return all_results



def find_fulfillments_violations(candidate, template, log):
    """
    For each trace in positional log give fulfilled and violated positions
    :param candidate:
    :param template:
    :param log:
    :return:
    """

    outp = apply_data_template_to_log(template, candidate, log)
    return outp




def get_data_snapshots(trace, fulfilled, violated):
    positive_snapshots = []
    negative_snapshots = []

    pos_locs = set(fulfilled)
    neg_locs = set(violated)

    current_snap = {}
    for i, event_data in enumerate(trace["data"]):
        for k, val in event_data.items():
            current_snap[k] = val

        if i in pos_locs:
            positive_snapshots.append(dict(current_snap))
        elif i in neg_locs:
            negative_snapshots.append(dict(current_snap))


    return positive_snapshots, negative_snapshots



class DRC:

    def create_sample(self, samples, features, label):
        features_data = []
        for smp_id, pos_act in samples:
            act_features = []
            act_features.append(smp_id)
            for feature in features:
                ft_type = feature[1]
                if feature in pos_act:
                    ft_val = pos_act[feature]
                    if ft_type == "boolean":
                        if ft_val == "true":
                            act_features.append(1)
                        elif ft_val == "false":
                            act_features.append(0)
                        else:
                            act_features.append(0)
                    elif ft_type == "literal":
                        act_features.append(ft_val)
                    elif ft_type == "continuous":
                        act_features.append(float(ft_val))
                    elif ft_type == "discrete":
                        act_features.append(int(ft_val))
                    else:
                        print("SHOULDNT BE HERE!")
                        raise Exception("Incorrect feature type in creation of samples")

                else:
                    if ft_type == "boolean":
                        act_features.append(0)
                    elif ft_type == "literal":
                        act_features.append("Missing")
                    elif ft_type == "continuous":
                        act_features.append(0)
                    elif ft_type == "discrete":
                        act_features.append(0)
                    else:
                        print("SHOULDNT BE HERE!")
                        raise Exception("Incorrect feature type in creation of samples")
            act_features.append(label)
            features_data.append(act_features)

        return features_data


    def create_data_aware_features(self, train_log, test_log, ignored):
        # given log
        # 0.0. Extract events

        # 1.1. Apriori mine events to be used for constraints

        # 2. Find declare constraints to be used, On a limited set of declare templates
        # 2.1. Find support for all positive and negative cases for the constraints
        # 2.2. Filter the constraints according to support
        # -- Encode the data
        # 3. Sort constraints according to Fisher score (or other metric)
        # 4. Pick the constraint with highest Fisher score.
        # 5. Refine the constraint with data
        # 5.1. Together with data, try to create a better rule.
        # ---- In this case, every node will become a small decision tree of its own!

        # 5.2. If the Fisher score of new rule is greater, change the current rule to a refined rule
        # --- Refined rule is - constraint + a decision rules / tree, learne
        # Reorder constraints for next level of decision tree .. It is exactly like Gini impurity or sth..

        # Get templates from fabrizios article

        """
        responded existence(A, B), data on A
        response(A, B), data on A
        precedence(A, B), data on B
        alternate response(A, B), data on A
        alternate precedence(A, B), data on B
        chain response(A,B), data on A
        chain precedence(A, B), data on B
        not resp. existence (A, B), data on A
        not response (A, B), data on A
        not precedence(A, B), data on B
        not chain response(A,B), data on A
        not chain precedence(A,B), data on B

        :param log:
        :param label:
        :return:
        """

        not_templates = ["not_responded_existence",
                         "not_precedence",
                         "not_response",
                         "not_chain_response",
                         "not_chain_precedence"]

        templates = ["alternate_precedence", "alternate_response", "chain_precedence", "chain_response",
                     "responded_existence", "response", "precedence"]


        inp_templates = templates + not_templates

        # play around with thresholds

        constraint_threshold = 0.1
        candidate_threshold = 0.1


        # Extract unique activities from log
        events_set = extract_unique_events_transformed(train_log)

        # Brute force all possible candidates
        candidates = [(event,) for event in events_set] + [(e1, e2) for e1 in events_set for e2 in events_set if
                                                           e1 != e2]

        # Count by class
        normal_count, deviant_count = count_classes(train_log)
        print("{} deviant and {} normal traces in train set".format(deviant_count, normal_count))
        ev_support_norm = int(normal_count * candidate_threshold)
        ev_support_dev = int(deviant_count * candidate_threshold)

        print("Filtering candidates by support")
        candidates = filter_candidates_by_support(candidates, train_log, ev_support_norm, ev_support_dev)
        print("Support filtered candidates:", len(candidates))

        constraint_support_dev = int(deviant_count * constraint_threshold)
        constraint_support_norm = int(normal_count * constraint_threshold)

        train_results = generate_train_candidate_constraints(candidates, inp_templates, train_log, constraint_support_norm,
                                                             constraint_support_dev, filter_t=True)



        test_results = generate_test_candidate_constraints(candidates, inp_templates, test_log, train_results)
        print("Candidate constraints generated")


        ## Given selected constraints, find fulfillments and violations for each of the constraint.
        ## In this manner build positive and negative samples for data

        X_train, y_train, feature_names, train_trace_names = transform_results_to_numpy(train_results, train_log)
        X_test, y_test, _, test_trace_names = transform_results_to_numpy(test_results, test_log)


        # Turn to pandas df
        train_df = pd.DataFrame(X_train, columns=feature_names, index=train_trace_names)

        train_df = train_df.transpose().drop_duplicates().transpose()

        # remove no-variance, constants
        train_df = train_df.loc[:, (train_df != train_df.iloc[0]).any()]

        X_train = train_df.values

        # Perform selection by Fisher

        scores = fisher_calculation(X_train, y_train)
        selected_ranks = fisher_score.feature_ranking(scores)

        threshold = 15
        #chosen = 500

        real_selected_ranks = []
        # Start selecting from selected_ranks until every trace is covered N times
        trace_remaining = dict()
        for i, trace_name in enumerate(train_df.index.values):
            trace_remaining[i] = threshold

        chosen = 0
        # Go from higher to lower
        for rank in selected_ranks:
            if len(trace_remaining) == 0:
                break
            chosen += 1
            # Get column
            marked_for_deletion = set()
            added = False
            for k in trace_remaining.keys():
                if train_df.iloc[k, rank] > 0:
                    if not added:
                        added = True
                        real_selected_ranks.append(rank)

                    trace_remaining[k] -= 1
                    if trace_remaining[k] <= 0:
                        marked_for_deletion.add(k)

            for k in marked_for_deletion:
                del trace_remaining[k]

        print("Constraints chosen {}".format(len(real_selected_ranks)))

        feature_names = train_df.columns[real_selected_ranks]

        print("Considered template count:", len(feature_names))
        train_df = train_df[feature_names]

        new_train_feature_names = []
        new_train_features = []

        new_test_feature_names = []
        new_test_features = []

        count=0

        for key in train_df.columns:

            count += 1
            #print(key)
            # Go over all and find with data
            template = key[0]
            candidate = key[1]

            # First have to find all locations of fulfillments
            outp_train = find_fulfillments_violations(candidate, template, train_log)
            outp_test = find_fulfillments_violations(candidate, template, test_log)

            # Take data snapshots on all fulfilled indices - positives samples
            # Take data snapshots on all unfulfilled indices - negative samples
            # Build a decision tree with fulfilled and unfulfilled samples
            train_positive_samples = []
            train_negative_samples = []

            test_positive_samples = []
            test_negative_samples = []

            for i, trace in enumerate(outp_train):
                fulfilled = trace[1]
                violated = trace[2]
                positive, negative = get_data_snapshots(train_log[i], fulfilled, violated)
                label = train_log[i]["label"]
                for s in positive:
                    train_positive_samples.append((s, label, i))
                for s in negative:
                    train_negative_samples.append((s, label, i))


            for i, trace in enumerate(outp_test):
                fulfilled = trace[1]
                violated = trace[2]
                positive, negative = get_data_snapshots(test_log[i], fulfilled, violated)
                label = train_log[i]["label"]

                for s in positive:
                    test_positive_samples.append((s, label, i))

                for s in negative:
                    test_negative_samples.append((s, label, i))

            # Get all where fulfilled only. Train on train_positive_samples vs Label of log
            ignored_features = set(ignored) # set([('Diagnose', 'literal')])

            collected_features = set()
            # Get all possible features for
            for pos_act, _, __ in train_positive_samples:
                for key2, val in pos_act.items():
                    collected_features.add(key2)

            for neg_act, _, __ in train_negative_samples:
                for key2, val in neg_act.items():
                    collected_features.add(key2)


            features = list(collected_features)

            # Keep only features of boolean, literal, continuous and discrete
            features = [feature for feature in features if feature[1] in set(["boolean", "continuous", "discrete", "literal"])]
            features = [feature for feature in features if feature[0] not in ignored_features]

            # collect positive and negative samples for finding data condition:
            positive_samples = [(sample[2], sample[0]) for sample in train_positive_samples if sample[1] == 1]
            negative_samples = [(sample[2], sample[0]) for sample in train_positive_samples if sample[1] == 0]

            pos_activations = [(sample[2], sample[0]) for sample in train_positive_samples]
            neg_activations = [(sample[2], sample[0]) for sample in train_negative_samples]

            feature_train_samples = self.create_sample(pos_activations, features, 1) + self.create_sample(neg_activations, features, 0)
            # Crete pos and neg samples
            pos_samples = self.create_sample(positive_samples, features, 1)
            neg_samples = self.create_sample(negative_samples, features, 0)
            features_data = pos_samples + neg_samples
            features_label = ["id"] + features + ["Label"]
            # one-hot encode literal features
            literal_features = [feature for feature in features if feature[1] == "literal"]

            # Extract positive test samples, where fulfillments where fulfilled
            train_df = pd.DataFrame(features_data, columns=features_label)
            test_pos_smpl = [(sample[2], sample[0]) for sample in test_positive_samples] # if sample[1] == 1]
            test_neg_smpl = [(sample[2], sample[0]) for sample in test_negative_samples] # if sample[1] == 0]

            pos_test_samples = self.create_sample(test_pos_smpl, features, 1)
            neg_test_samples = self.create_sample(test_neg_smpl, features, 0)
            test_features_data = pos_test_samples + neg_test_samples

            feature_train_df = pd.DataFrame(feature_train_samples, columns=features_label)
            test_df = pd.DataFrame(test_features_data, columns=features_label)
            train_df.pop("id")
            train_ids = feature_train_df.pop("id")
            test_ids = test_df.pop("id")


            # Possible values for each literal value is those in train_df or missing

            if len(literal_features) > 0:
                for selection in literal_features:
                    train_df[selection] = pd.Categorical(train_df[selection])
                    test_df[selection] = pd.Categorical(test_df[selection])
                    feature_train_df[selection] = pd.Categorical(feature_train_df[selection])
                    le = LabelEncoder()

                    le.fit(list(test_df[selection]) + list(feature_train_df[selection]))
                    classes = le.classes_
                    train_df[selection] = le.transform(train_df[selection])
                    test_df[selection] = le.transform(test_df[selection])
                    feature_train_df[selection] = le.transform(feature_train_df[selection])

                    ohe = OneHotEncoder(categories="auto") # Remove this for server.
                    ohe.fit(np.concatenate((test_df[selection].values.reshape(-1, 1),
                                            feature_train_df[selection].values.reshape(-1, 1)), axis=0),)

                    train_transformed = ohe.transform(train_df[selection].values.reshape(-1, 1)).toarray()
                    test_transformed = ohe.transform(test_df[selection].values.reshape(-1, 1)).toarray()
                    feature_train_transformed = ohe.transform(feature_train_df[selection].values.reshape(-1, 1)).toarray()

                    dfOneHot = pd.DataFrame(train_transformed,
                                            columns=[(selection[0] + "_" + classes[i], selection[1]) for i in
                                                     range(train_transformed.shape[1])])
                    train_df = pd.concat([train_df, dfOneHot], axis=1)
                    train_df.pop(selection)
                    dfOneHot = pd.DataFrame(test_transformed,
                                            columns=[(selection[0] + "_" + classes[i], selection[1]) for i in
                                                     range(train_transformed.shape[1])])
                    test_df = pd.concat([test_df, dfOneHot], axis=1)
                    test_df.pop(selection)

                    dfOneHot = pd.DataFrame(feature_train_transformed,
                                            columns=[(selection[0] + "_" + classes[i], selection[1]) for i in
                                                     range(train_transformed.shape[1])])
                    feature_train_df = pd.concat([feature_train_df, dfOneHot], axis=1)
                    feature_train_df.pop(selection)

            data_dt = DecisionTreeClassifier(max_depth=3)
            y_train = train_df.pop("Label")
            train_data = train_df.values

            y_test = test_df.pop("Label")
            data_dt.fit(train_data, y_train)

            y_train_new = feature_train_df.pop("Label")
            feature_train_data = feature_train_df.values

            train_predictions = data_dt.predict(feature_train_data)
            test_predictions = data_dt.predict(test_df.values)

            train_fts = feature_train_df.columns
            # Go through all traces again
            # Save decision trees here. For later interpretation
            feature_train_df["id"] = train_ids
            test_df["id"] = test_ids

            feature_train_df["prediction"] = train_predictions
            test_df["prediction"] = test_predictions

            # Check for which activations the data condition holds. Filter everything else out.

            feature_train_df["Label"] = y_train_new
            test_df["Label"] = y_test

            new_train_feature = []
            for i, trace in enumerate(outp_train):
                # Get from train_df by number
                trace_id = i
                freq = trace[0]

                # Find all related to the id

                if freq == 0:
                    # vacuous case, no activations, will be same here.
                    new_train_feature.append(0)
                else:
                    # Previous violation case
                    # Find samples related to trace
                    samples = feature_train_df[feature_train_df.id == trace_id]
                    # Find samples related for which data condition holds
                    samples = samples[samples.prediction == 1]
                    # Count number of positive and negative labels
                    positive = samples[samples.Label == 1].shape[0]
                    negative = samples[samples.Label == 0].shape[0]

                    if negative > 0:
                        new_train_feature.append(-1)
                    else:
                        new_train_feature.append(positive)

            new_test_feature = []

            for i, trace in enumerate(outp_test):
                # Get from train_df by number
                trace_id = i
                freq = trace[0]

                # Find all related to the id

                if freq == 0:
                    # vacuous case, no activations, will be same here.
                    new_test_feature.append(0)
                else:
                    # Previous violation case
                    # Find samples related to trace
                    samples = test_df[test_df.id == trace_id]
                    # Find samples related for which data condition holds
                    samples = samples[samples.prediction == 1]
                    # Count number of positive and negative activations
                    positive = samples[samples.Label == 1].shape[0]
                    negative = samples[samples.Label == 0].shape[0]

                    if negative > 0:
                        new_test_feature.append(-1)
                    else:
                        new_test_feature.append(positive)

            # Find all activatio

            count_fulfilled_train = sum(1 for i in new_train_feature if i > 0)
            count_fulfilled_test = sum(1 for i in new_test_feature if i > 0)

            if count_fulfilled_train > 0 and count_fulfilled_test > 0:
                # only then add new feature..
                new_train_features.append(new_train_feature)
                new_train_feature_names.append(template + ":({},{}):Data".format(candidate[0], candidate[1]))

                new_test_features.append(new_test_feature)
                new_test_feature_names.append(template + ":({},{}):Data".format(candidate[0], candidate[1]))

                # Save decision tree
                if True:
                    export_graphviz(data_dt, out_file="sample_dwd_trees/outputfile_{}.dot".format(str(key)),
                                    feature_names=list(map(str, train_fts)))

        return new_train_feature_names, new_train_features, new_test_feature_names, new_test_features



def data_declare_main(inp_folder, log_name, ignored):

    drc = DRC()
    log = read_XES_log(log_name)

    # Transform log into suitable data structures
    transformed_log = xes_to_data_positional(log)

    train_log, test_log = split_log_train_test(transformed_log, 0.8)
    #print(train_log[0])

    train_case_ids = [tr["name"] for tr in train_log]
    test_case_ids = [tr["name"] for tr in test_log]

    train_names, train_features, test_names, test_features = drc.create_data_aware_features(train_log, test_log, ignored)

    train_dict = {}
    test_dict = {}
    for i, tf in enumerate(train_features):
        train_dict[train_names[i]] = tf

    for i, tf in enumerate(test_features):
        test_dict[test_names[i]] = tf

    train_df = pd.DataFrame.from_dict(train_dict)
    test_df = pd.DataFrame.from_dict(test_dict)


    #train_df = pd.DataFrame(train_features, columns=train_names)
    #test_df = pd.DataFrame(test_features, columns=test_names)

    #print(train_names)
    # add Case_ID

    train_df["Case_ID"] = train_case_ids
    test_df["Case_ID"] = test_case_ids

    train_df.to_csv(inp_folder + "/dwd_train.csv", index=False)
    test_df.to_csv(inp_folder + "/dwd_test.csv", index=False)


def move_dwd_files(inp_folder, output_folder, split_nr):
    source = inp_folder # './baselineOutput/'
    dest1 = './' + output_folder + '/split' + str(split_nr) + "/dwd/"
    files = os.listdir(source)
    for f in files:
        shutil.move(source + f, dest1)


def run_declare_with_data(log_path, settings, results_folder):
    for logNr in range(5):
        logPath = log_path.format(logNr + 1)
        folder_name = "./dwdOutput/"
        if not os.path.exists(folder_name):
            os.makedirs(folder_name)

        ignored = settings["ignored"]

        data_declare_main(folder_name, logPath, ignored)
        move_dwd_files(folder_name, results_folder, logNr + 1)



if __name__ == "__main__":
    for log_nr in [1,2,3,4,5]:
        #log_path = "logs/sepsis_tagged_er.xes"
        log_path = "EnglishBPI/EnglishBPIChallenge2011_tagged_cc_{}.xes".format(log_nr)
        ignored = ["time:timestamp", "concept: name", "Label", "Start date", "End date", "Diagnosis", "Diagnosis code",
                   "Diagnosis Treatment", "Combination ID", "Treatment code", "Activity code"]


        data_declare_main(log_path, ignored)
