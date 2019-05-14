"""
Mining features based on frequent patterns, experimental for later..
Mines based on frequent subsequences
not used in thesis
"""

from itertools import islice
from prefixspan import PrefixSpan
from time import time
from opyenxes.data_in import XUniversalParser

from itertools import combinations
from collections import Counter

def read_XES_log(path):
    tic = time()

    print("Parsing log")
    with open(path) as log_file:
        log = XUniversalParser().parse(log_file)[0]  # take first log from file

    toc = time()

    print("Log parsed, took {} seconds..".format(toc - tic))

    return log


def subsequence_counts_3(sequences):
    return Counter(seq[i:j]
                   for seq in map(''.join, sequences)
                   for i, j in combinations(range(len(seq) + 1), 2))


def window(seq, n=2):
    "Returns a sliding window (of width n) over data from the iterable"
    "   s -> (s0,s1,...s[n-1]), (s1,s2,...,sn), ...                   "

    it = iter(seq)

    result = tuple(islice(it, n))

    if len(result) == n:
        yield result
    for elem in it:
        result = result[1:] + (elem,)
        yield result



def mine_frequent_span(log):
    input = []

    different_events = set()

    for trace in log:
        trace_events = []
        for event in trace:
            event_attribs = event.get_attributes()
            event_name = str(event_attribs["concept:name"])
            if "lifecycle:transition" in event_attribs:
                event_name += "-" + str(event_attribs["lifecycle:transition"])
            trace_events.append(event_name)
            different_events.add(event_name)
        input.append(trace_events)


    # Encode input
    encoding = {

    }

    decoding = {

    }
    for i, event in enumerate(different_events):
        encoding[event] = i
        decoding[i] = event

    # Encode traces
    minimum_size = 5

    encoded = [[encoding[event] for event in sublist] for sublist in input]
    ps = PrefixSpan(encoded)

    outputs = ps.topk(10000)

    decoded_output = list(reversed(sorted([(sublist[0], [decoding[output] for output in sublist[1]]) for sublist in outputs], key=lambda x: x[0])))

    #print(decoded_output)
    to_file = "\n".join(map(str, decoded_output))

    with open("frequent_subs.txt", "w") as f:
        f.write(to_file)


def mine_frequent_patterns(log):
    input = []

    different_events = set()

    for trace in log:
        trace_events = []
        for event in trace:
            event_attribs = event.get_attributes()
            event_name = str(event_attribs["concept:name"])
            if "lifecycle:transition" in event_attribs:
                event_name += "-" + str(event_attribs["lifecycle:transition"])
            trace_events.append(event_name)
            different_events.add(event_name)
        input.append(trace_events)


    # Encode input
    encoding = {

    }

    decoding = {

    }
    for i, event in enumerate(different_events):
        encoding[event] = i
        decoding[i] = event

    # Encode traces
    minimum_size = 5

    encoded = [[encoding[event] for event in sublist] for sublist in input]

    threshold = int(0.3 * len(log))
    outputs = []

    for length in range(3, 9):
        counts = Counter()
        for parts in encoded:
            # Only once per log!
            existences = set()
            for sp in window(parts, length):
                existences.add('-'.join(map(str, sp)))

            counts.update(existences)

        for k,v in counts.most_common(10):
            outputs.append((v, map(int, k.split("-"))))

    decoded_output = list(reversed(sorted([(sublist[0], [decoding[output] for output in sublist[1]]) for sublist in outputs], key=lambda x: x[0])))

    print(decoded_output)
    to_file = "\n".join(map(str, decoded_output))

    with open("frequent_patterns.txt", "w") as f:
        f.write(to_file)

def mine_sepsis_fp():
    path = "logs/sepsis.xes"


    log = read_XES_log(path)

    #mine_frequent_patterns(log)

    mine_frequent_span(log)

    #shuffle(log)
    #with open("logs/sepsis_constraint_tagged.xes", "w") as file:
    #    XesXmlSerializer().serialize(log, file)

