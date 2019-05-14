"""
Extraction of rule statistics from text output of rules.
Average length of rules, average count of rules in experiment.

"""

import numpy as np

jrip_rules = "rules_jrip.txt"
dt_rules = "rules_dt.txt"

from collections import defaultdict

def get_jriprules():
    with open(jrip_rules) as f:
        current_experiment = ""
        rule_lengths = []
        rule_counts = []
        res = defaultdict(list)
        for line in f:
            if line.startswith("Working on"):
                if current_experiment != "":
                    exp = current_experiment.split()[2]
                    coverage = current_experiment.split()[4]
                    res[(exp, coverage)].append((np.mean(rule_counts), np.mean(rule_lengths)))
                    avg_rule_lengths = []
                    rule_lengths = []
                    rule_counts = []
                current_experiment = line.strip()

            elif line.startswith("Number of Rules"):
                loc_hy = line.find(":")
                rules_count = int(line[loc_hy + 1:])

                rule_counts.append(rules_count)
            elif "=>" in line:
                if line[1:3] == "=>":
                    rule_length = 0
                else:
                    rule_length = line.count(") and (") + 1
                # print(line)
                # print(rule_length)
                rule_lengths.append(rule_length)

        exp = current_experiment.split()[2]
        coverage = current_experiment.split()[4]
        res[(exp, coverage)].append((np.mean(rule_lengths), np.mean(rule_counts)))

        coverages = [5, 15, 25]
        # """
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
        experiments = ["bpi2011_cc", "bpi2011_m13", "bpi2011_m16", "bpi2011_t101"]

        for exp in experiments:
            print(exp)
            for coverage in coverages:
                for a, b in res[(exp, str(coverage))]:
                    print(a, b)
                print()


def get_dtrules():
    with open(dt_rules) as f:
        current_experiment = ""
        rule_lengths = []
        rule_counts = []
        res = defaultdict(list)
        number_of_rules = 0
        for line in f:
            if line.startswith("Working on"):
                if current_experiment != "":
                    exp = current_experiment.split()[2]
                    coverage = current_experiment.split()[4]
                    rule_counts.append(number_of_rules + 1) # +1 for default rules in each
                    # Add one zero-rule for each split.. accounts for default rule
                    rule_lengths += [0, 0, 0, 0, 0]
                    res[(exp, coverage)].append((np.mean(rule_counts), np.mean(rule_lengths)))
                    rule_lengths = []
                    rule_counts = []
                current_experiment = line.strip()
            elif line.startswith("Split 1"):
                continue
            elif line.startswith("Split "):
                rule_counts.append(number_of_rules + 1) # +1 for default rule
                number_of_rules = 0
            elif "=>" in line:
                number_of_rules += 1
                rule_length = line.count(") and (") + 1
                rule_lengths.append(rule_length)

        exp = current_experiment.split()[2]
        coverage = current_experiment.split()[4]
        res[(exp, coverage)].append((np.mean(rule_lengths), np.mean(rule_counts)))

        coverages = [5, 15, 25]
        experiments = ["bpi2011_cc", "bpi2011_m13", "bpi2011_m16", "bpi2011_t101"]

        for exp in experiments:
            print(exp)
            for coverage in coverages:
                for a, b in res[(exp, str(coverage))]:
                    print(a, b)
                print()


get_dtrules()