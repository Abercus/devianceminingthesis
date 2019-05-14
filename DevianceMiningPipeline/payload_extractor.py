"""
Pure data payload extraction method and encoding.
"""


from deviancecommon import read_XES_log, split_log_train_test

from opyenxes.model import XAttributeBoolean, XAttributeLiteral, XAttributeTimestamp, XAttributeDiscrete, \
    XAttributeContinuous
from collections import defaultdict

import pandas as pd
import os, shutil

from sklearn.preprocessing import LabelEncoder, OneHotEncoder

def get_attribute_type(val):
    if isinstance(val, XAttributeLiteral.XAttributeLiteral):
        return "literal"
    elif isinstance(val, XAttributeBoolean.XAttributeBoolean):
        return "boolean"
    elif isinstance(val, XAttributeDiscrete.XAttributeDiscrete):
        return "discrete"
    elif isinstance(val, XAttributeTimestamp.XAttributeTimestamp):
        return "timestamp"
    elif isinstance(val, XAttributeContinuous.XAttributeContinuous):
        return "continuous"


def handle_first(value, options, trace_data):
    """
    Handles extraction of first from event...
    :param value: xesEvent's value
    :param options: is specific to event, what options...
    :param trace_data: extracted data
    :return:
    """
    if options not in trace_data:
        trace_data[options] = str(value)


def handle_last(value, options, trace_data):
    """
    :param value:
    :param options:
    :param trace_data:
    :return:
    """
    trace_data[options] = str(value)


def handle_min(value, options, trace_data):
    """

    :param value:
    :param options:
    :param trace_data:
    :return:
    """
    # Has to be float or integer
    att_type = get_attribute_type(value)
    assert (att_type in ["continuous, discrete"])

    if att_type == "continuous":
        new_val = float(str(value))
    elif att_type == "discrete":
        new_val = int(str(value))

    if options in trace_data:
        # Already in traces
        new_val = min(new_val, trace_data[options][1])

    trace_data[options] = new_val


def handle_max(value, options, trace_data):
    """

    :param value:
    :param options:
    :param trace_data:
    :return:
    """
    # Has to be float or integer
    att_type = get_attribute_type(value)
    assert (att_type in ["continuous", "discrete"])

    if att_type == "continuous":
        new_val = float(str(value))
    elif att_type == "discrete":
        new_val = int(str(value))

    if options in trace_data:
        # Already in traces
        new_val = max(new_val, trace_data[options][1])

    trace_data[options] = new_val


def handle_mean(value, options, trace_data):
    """

    :param value:
    :param options:
    :param trace_data:
    :return:
    """
    # For mean calculations - keep summed score and count. Calculated at the end
    att_type = get_attribute_type(value)
    assert (att_type in ["continuous, discrete"])

    if att_type == "continuous":
        new_val = float(str(value))
    elif att_type == "discrete":
        new_val = int(str(value))

    if options in trace_data:
        # Already in traces
        new_val = new_val + trace_data[options][1]
        trace_data[options] = (trace_data[options][0] + 1, new_val)
    else:
        trace_data[options] = (1, new_val)


def handle_count(value, options, trace_data):
    """
    count of certain attribute.

    If literal: Color: Blue, Color: Red ... build output

    trace_data[options] = {
        blue: 1,
        red: 2

    }

    Available for: boolean, literal
    :param value:
    :param options:
    :param trace_data:
    :return:
    """

    options = (options[0], options[1] + "_" + str(value), options[2], options[3])
    #options[1] += "_" + str(value)
    if options not in trace_data:
        trace_data[options] = 0

    trace_data[options] += 1


def calculate_mean(value, options, trace_data):
    """
    After collecting required information for means we need to calc it
    Available for continuous and float
    :param value:
    :param options:
    :param trace_data:
    :return:
    """
    pass


def handle_event_time_per_event(value, options, trace_data):
    """
    Divide total time taken by number of events.
    :param value:
    :param options:
    :param trace_data:
    :return:
    """
    pass


class PayloadExtractor:

    def __init__(self):
        self.mode = set(["literal", "boolean", "discrete"])
        #self.mode = set(["literal"])
        self.event_ignored = set(["lifecycle:transition", "concept:name", "time:timestamp", "Diagnose", "Age", "Label"])
        self.trace_ignored = set(["concept:name", "org:group", "time:timestamp", "lifecycle:transition", "Label",
                                  "Diagnosis code", "Treatment code", "Diagnosis"])
        self.event_attribs = set(["org:resource"])
        self.events_as_trace = set(["ER Registration"])
        self.event_names_ignored = set(["Return ER"])

    def get_attribute_type(self, val):
        if isinstance(val, XAttributeLiteral.XAttributeLiteral):
            return "literal"
        elif isinstance(val, XAttributeBoolean.XAttributeBoolean):
            return "boolean"
        elif isinstance(val, XAttributeDiscrete.XAttributeDiscrete):
            return "discrete"
        elif isinstance(val, XAttributeTimestamp.XAttributeTimestamp):
            return "timestamp"
        elif isinstance(val, XAttributeContinuous.XAttributeContinuous):
            return "continuous"

    def add_trace_attribs(self, pl, attribs):
        ignored = set()
        for k, v in attribs.items():
            if k not in self.trace_ignored:
                pl[k] = {
                    "type": self.get_attribute_type(v),
                    "value": str(v)
                }
            else:
                ignored.add(k)
        return ignored

    def add_event_attribs(self, pl, attribs):

        for k, v in attribs.items():
            if k not in self.event_ignored:
                name = str(attribs["concept:name"])
                att_type = self.get_attribute_type(v)
                # Check if attribute type in mode
                if att_type not in self.mode:
                    continue

                if k not in pl:
                    pl[k] = {
                        "type": att_type
                    }

                if "data" not in pl[k]:
                    pl[k]["data"] = []

                pl[k]["data"].append(
                    {"name": name, "value": str(v)}
                )

    def get_overview_of_data(self, log, settings=None):
        """
        For each event:
        Check what different kind of data they have, and how often does it appear in data.

        :return:
        """

        # 1. Go through log, collect all event attributes and trace attributes

        # 2. Give statistics of all attributes - nans, etc.

        # 3. possible to mark attributes to be event-agnostic - such as org:resource, time..
        # 4. Or to be strictly related to trace, which uses summaries - ex how many times did that org:resource appear
        # ... different ways to aggregate: ex. sum up some values etc.. Make config file for aggregation

        """
        Config file can consist of:
        -- trace/event attributes to be ignored
        -- event attributes for aggregation, separately or for both
        ---- Aggregation - 
        ----- For numbers example sum, mean, max, min, std
        ----- For other: count of each label - ex. org:resource A: 10 
        ----- Count of missing values
        -- By default all attributes are strictly related to their event
        ---- ex, we have event name "Submit" with attribute resource and value 50, then we get event:submit:resource: 50
        -- Option to have extra statistics per trace
        ---- Total time taken, count of events in trace
        --- TODO: How to handle missing values for each!
        ----- DEFAULT behavior: for literal, make new class "missing", for discrete/continuous: mean value (give warning!)! 
        """
        # Goal of this tool is just to extract features.

        if settings == None:
            settings = {
                "trace_ignored": set(["time:timestamp", "concept:name"]),
                "event_ignored": set(["time:timestamp", "lifecycle:transition"]),
                #"trace_extra": [("time", "days"), ("length",)],
                "trace_extra": [("length",)],
                "event": {},
                "attribute_aggregate": {"org:group" : ["count", "first"]},
                "event_default": ["first"]  # If there are several events, with same name and same kind of payload,
                #  then how to handle it. first means to keep first one, last to keep the last one. If there is not described..
            }

        mode = "overview"

        log_data = []

        for trace in log:
            # 1. step
            trace_data = {}
            """
            trace_data = {
                concept:name : name,
                first: first_events_attribute_value,
                last: last_events_attribute_value,
                length: length_of_trace
            
            }
            
            """

            # Handle trace attributes
            trace_attribs = trace.get_attributes()
            trace_name = str(trace_attribs["concept:name"])
            trace_data["concept:name"] = trace_name

            for attrib_name, value in trace_attribs.items():
                # handle trace attributes
                attrib_type = get_attribute_type(value)
                if attrib_name == "concept:name" or attrib_name in settings["trace_ignored"]:
                    continue

                trace_data[(None, "trace" + ":" + attrib_name, "first", attrib_type)] = str(value)


            # handle trace extra attribs

            for option in settings["trace_extra"]:
                if option[0] == "length":
                    trace_data["length"] = len(trace)
                elif option[0] == "time":
                    time_option = option[1]
                    pass # Calculate based on time_option, if days then calculate how many days between latest and first event

            for event in trace:

                attribs = event.get_attributes()
                event_name = str(attribs["concept:name"])

                for attrib_name, value in attribs.items():
                    if attrib_name == "concept:name" or attrib_name in settings["event_ignored"]:
                        continue
                    att_type = self.get_attribute_type(value)

                    # Switch for different handling based on settings
                    # First check if there is described option for the attribute
                    if attrib_name in settings["event"]:
                        for option in settings["event"][attrib_name]:
                            # Attribute-name specific ..
                            # (event name (None, if aggregate), attrib name, option type.)
                            target = (event_name, attrib_name, option, att_type)

                            if option == "first":
                                if att_type in ("continuous", "discrete", "literal", "boolean"):
                                    handle_first(value, target, trace_data)
                            elif option == "last":
                                if att_type in ("continuous", "discrete", "literal", "boolean"):
                                    handle_last(value, target, trace_data)

                            # TODO: MIN, MAX, MEAN, can only be calculated for continuous and discrete values
                            elif option == "min":
                                if att_type in ("continuous", "discrete"):
                                    handle_min(value, target, trace_data)

                            elif option == "max":
                                if att_type in ("continuous", "discrete"):
                                    handle_max(value, target, trace_data)

                            elif option == "mean":
                                if att_type in ("continuous", "discrete"):
                                    handle_mean(value, target, trace_data)

                            # TODO: COUNT IS BY DEFAULT CALCULATED FOR LITERAL VALUES ONLY
                            # COULD ALSO BE USED FOR INTEGER AND BOOLEAN

                            elif option == "count":
                                if att_type == "literal":
                                    handle_count(value, target, trace_data)

                    elif attrib_name in settings["attribute_aggregate"]:
                        for option in settings["attribute_aggregate"][attrib_name]:
                            # (event name (None, if aggregate), attrib name, option type.)
                            target = (None, attrib_name, option, att_type)

                            if option == "first":
                                if att_type in ("continuous", "discrete", "literal", "boolean"):
                                    handle_first(value, target, trace_data)
                            elif option == "last":
                                if att_type in ("continuous", "discrete", "literal", "boolean"):
                                    handle_last(value, target, trace_data)

                            # TODO: MIN, MAX, MEAN, can only be calculated for continuous and discrete values
                            elif option == "min":
                                if att_type in ("continuous", "discrete"):
                                    handle_min(value, target, trace_data)

                            elif option == "max":
                                if att_type in ("continuous", "discrete"):
                                    handle_max(value, target, trace_data)

                            elif option == "mean":
                                if att_type in ("continuous", "discrete"):
                                    handle_mean(value, target, trace_data)

                            # TODO: COUNT IS BY DEFAULT CALCULATED FOR LITERAL VALUES ONLY
                            # COULD ALSO BE USED FOR INTEGER AND BOOLEAN

                            elif option == "count":
                                if att_type == "literal":
                                    handle_count(value, target, trace_data)
                    else:
                        # Do default for event - aggregations by event attribute name
                        for option in settings["event_default"]:
                            # (event name (None, if aggregate), attrib name, option type.)
                            target = (None, attrib_name, option, att_type)

                            if option == "first":
                                if att_type in ("continuous", "discrete", "literal", "boolean"):
                                    handle_first(value, target, trace_data)
                            elif option == "last":
                                if att_type in ("continuous", "discrete", "literal", "boolean"):
                                    handle_last(value, target, trace_data)

                            # TODO: MIN, MAX, MEAN, can only be calculated for continuous and discrete values
                            elif option == "min":
                                if att_type in ("continuous", "discrete"):
                                    handle_min(value, target, trace_data)

                            elif option == "max":
                                if att_type in ("continuous", "discrete"):
                                    handle_max(value, target, trace_data)

                            elif option == "mean":
                                if att_type in ("continuous", "discrete"):
                                    handle_mean(value, target, trace_data)

                            # TODO: COUNT IS BY DEFAULT CALCULATED FOR LITERAL VALUES ONLY
                            # COULD ALSO BE USED FOR INTEGER AND BOOLEAN

                            elif option == "count":
                                if att_type == "literal":
                                    handle_count(value, target, trace_data)

            # Calculate mean, other statistics that cant be calculated online
            # Time taken for trace - (days, hours, seconds, minutes... depends on configuration)
            # Length of the trace, number of events
            for key, val in trace_data.items():
                if key[2] == "mean":
                    trace_data[key] = val[1] / val[0]

            log_data.append(trace_data)

        return log_data

    def extract_trace_payload(self, trace, mode=None):
        """
        Trace_payload = {
            "payload_name": {
                "type": "literal",
                "data": data... integer, literal, boolean
            }
        }

        extracted_event_data = {
            "payload_name": {
                "type": "literal",
                "data": [{"name": name, val: value }, {...}
                ]
            }
        }
        :param trace:
        :param mode:
        :return:
        """
        trace_payload = {}
        trace_attribs = trace.get_attributes()
        self.add_trace_attribs(trace_payload, trace_attribs)

        extracted_event_data = defaultdict(dict)

        # For events
        for event in trace:
            attribs = event.get_attributes()
            # If the event attributes should be handled as trace attributes
            #print(str(attribs["concept:name"]))
            if str(attribs["concept:name"]) in self.event_names_ignored:
                continue
            if str(attribs["concept:name"]) in self.events_as_trace:
                ignored = self.add_trace_attribs(trace_payload, attribs)

                # Check if any ignored goes to event data
                pass_on = {}
                for ig in ignored:
                    pass_on[ig] = attribs[ig]

                self.add_event_attribs(extracted_event_data, pass_on)
            else:
                self.add_event_attribs(extracted_event_data, attribs)

        return {
            "trace": trace_payload,
            "events": extracted_event_data
        }

    def transform_discrete_payload(self, payloads, discrete_trace_inp, discrete_event_inp):
        if len(discrete_trace_inp) == 0 and len(discrete_event_inp) == 0:
            return None

        all_results = []
        # Handle if there are of literal type trace payload
        for payload in payloads:
            trace_payload = payload["trace"]
            payload_result = []
            for k, v in discrete_trace_inp:
                if k in trace_payload:
                    payload_result.append((k, int(trace_payload[k]["value"])))
                else:
                    payload_result.append((k, None))
            event_payload = payload["events"]
            for k, v in discrete_event_inp:
                counts = defaultdict(int)
                for t in v:
                    counts[t] = 0
                for data in event_payload[k]["data"]:
                    counts[data["value"]] += 1

                for cl in v:
                    payload_result.append((k + "_" + cl, counts[cl]))

            all_results.append(payload_result)

        return all_results


    def transform_literal_payload(self, payloads, literal_trace_inp, literal_event_inp):
        if len(literal_trace_inp) == 0 and len(literal_event_inp) == 0:
            return None

        all_results = []

        # Handle if there are of literal type trace payload
        for payload in payloads:
            trace_payload = payload["trace"]
            payload_result = []
            for k, v in literal_trace_inp:
                possible_outputs = set(v) # Have to check if it exists in train data..
                if k in trace_payload and trace_payload[k]["value"] in possible_outputs:
                    payload_result.append((k, trace_payload[k]["value"]))
                else:
                    payload_result.append((k, None))

            event_payload = payload["events"]
            for k, v in literal_event_inp:
                counts = defaultdict(int)
                possible_outputs = set(v)
                for t in v:
                    counts[t] = 0
                for data in event_payload[k]["data"]:
                    if data["value"] in possible_outputs:
                        counts[data["value"]] += 1

                for cl in v:
                    payload_result.append((k+"_"+cl, counts[cl]))

            all_results.append(payload_result)

        return all_results


    def transform_boolean_payload(self, payloads, bool_trace_inp, bool_event_inp):

        if len(bool_trace_inp) == 0 and len(bool_event_inp) == 0:
            return None

        all_results = []

        if len(bool_trace_inp) > 0 and len(bool_event_inp) > 0:
            pass
        elif len(bool_trace_inp) > 0:
            for payload in payloads:
                trace_payload = payload["trace"]
                payload_result = []
                for k, v in bool_trace_inp:
                    if k in trace_payload and trace_payload[k]["value"] in ("true", "false"):
                        payload_result.append((k, True if trace_payload[k]["value"] == "true" else False))
                    else:
                        payload_result.append((k, None))

                all_results.append(payload_result)

        elif len(bool_event_inp) > 0:
            pass

        return all_results

    def transform_continuous_payload(self, payloads, continuous_trace_inp, continuous_event_inp):
        if len(continuous_event_inp) == 0 and len(continuous_trace_inp) == 0:
            return None

        all_results = []
        # Handle if there are of literal type trace payload
        for payload in payloads:
            trace_payload = payload["trace"]
            payload_result = []
            for k, v in continuous_trace_inp:
                if k in trace_payload:
                    payload_result.append((k, int(trace_payload[k]["value"])))
                else:
                    payload_result.append((k, None))
            event_payload = payload["events"]
            for k, v in continuous_event_inp:
                counts = defaultdict(int)
                for t in v:
                    counts[t] = 0
                for data in event_payload[k]["data"]:
                    counts[data["value"]] += 1

                for cl in v:
                    payload_result.append((k + "_" + cl, counts[cl]))

            all_results.append(payload_result)

        return all_results





def switch_mode(stripped):

    mode = None


    if stripped == "--TRACE IGNORED--":
        mode = "trace_ignored"
    elif stripped == "--EVENT IGNORED--":
        mode = "event_ignored"
    elif stripped == "--TRACE EXTRA--":
        mode = "trace_extra"
    elif stripped == "--EVENT--":
        mode = "event"
    elif stripped == "--AGGREGATED ATTRIBUTES--":
        mode = "attribute_aggregate"
    elif stripped == "--EVENT DEFAULT--":
        mode = "event_default"
    elif stripped == "--MISSING--":
        mode = "missing"
    elif stripped == "--ONE HOT ENCODING--":
        mode = "one_hot_encode"

    return mode

def settings_from_cfg(settings_file):

    settings = {
        "trace_ignored": set(),
        "event_ignored": set(),
        "trace_extra": [],
        "event": {},
        "attribute_aggregate": {},
        "event_default": [],
        "missing": {},
        "one_hot_encode" : set()

    }
    modes = set(["--TRACE IGNORED--", "--EVENT IGNORED--", "--TRACE EXTRA--",
                 "--EVENT--", "--AGGREGATED ATTRIBUTES--", "--EVENT DEFAULT--",
                 "--MISSING--", "--ONE HOT ENCODING--"])

    mode = None
    with open(settings_file, "r") as f:

        for line in f:
            stripped = line.strip()
            if stripped in modes:
                mode = switch_mode(stripped)
            else:
                if mode == "trace_ignored" or mode == "event_ignored":
                    if len(stripped) > 0:
                        settings[mode].add(stripped)
                elif mode == "trace_extra":
                    if len(stripped) > 0:
                        splits = stripped.split("|")
                        if len(splits) == 1:
                            settings[mode].append((splits[0],))
                        elif len(splits) == 2:
                            settings[mode].append((splits[0], splits[1]))

                elif mode == "event":
                    # Specific event-related events.. Separation by events
                    pass
                elif mode == "attribute_aggregate":
                    if len(stripped) > 0:
                        splits = stripped.split("|")
                        attribute = splits[0]
                        aggregates = splits[1].split(",")
                        settings[mode][attribute] = aggregates
                elif mode == "event_default":
                    if len(stripped) > 0:
                        settings[mode].append(stripped)
                elif mode == "missing":
                    if len(stripped) > 0:
                        splits = stripped.split("|")
                        settings[mode][splits[0]] = splits[1]
                elif mode == "one_hot_encode":
                    if len(stripped) > 0:
                        splits = stripped.split("|")
                        settings[mode].add((splits[0], splits[1]))

    return settings


def handle_data(k, trace_data, trace_df_data, settings):
    if k not in trace_data:
        if k[3] == "boolean":
            if "boolean" in settings["missing"]:
                mode = settings["missing"]["boolean"]
                if mode == "false":
                    trace_df_data.append(0)
                elif mode == "true":
                    trace_df_data.append(1)
                else:
                    trace_df_data.append(None)
        elif k[3] == "literal":
            mode = settings["missing"]["literal"]
            if k[2] == "count":
                trace_df_data.append(0)
            elif mode == "missing":
                trace_df_data.append("missing")
            else:
                trace_df_data.append(None)
        elif k[3] == "discrete" or k[3] == "continuous":
            mode = settings["missing"]["numeric"]
            if mode == "mean":
                # imput mean! -- TODO: Need to do this later, after all extracted
                trace_df_data.append(None)
            elif mode == "0":
                trace_df_data.append(0)
            else:
                trace_df_data.append(None)
        else:
            print("ERROR! Shouldn't be here, not implemented for trace {}".format(k))
    else:
        if k[3] == "boolean":
            if trace_data[k] == "true":
                trace_df_data.append(1)
            elif trace_data[k] == "false":
                trace_df_data.append(0)
            else:
                trace_df_data.append(None)
        elif k[3] == "literal":
            trace_df_data.append(trace_data[k])
        elif k[3] == "discrete":
            trace_df_data.append(int(trace_data[k]))
        elif k[3] == "continuous":
            trace_df_data.append(float(trace_data[k]))
        elif k == "length":
            trace_df_data.append(int(trace_data[k]))
        elif k == "concept:name":
            trace_df_data.append(trace_data[k])
        else:
            print("ERROR! Not implemented for {}".format(k))
            raise Exception("Not implemented! Add to ignore in config file!")


def build_dataframes(train_data, test_data, settings):
    # Booleans will be false, true or NA (missing), missing if not given by settings

    train_df_data = []
    test_df_data = []

    # Collect all different keys from all train data.

    all_keys = set()

    for trace_data in train_data:
        keys = trace_data.keys()
        for key in keys:
            all_keys.add(key)

    # Build names from keys
    names = []

    # columns to one_hot_encode
    selected_one_hot = set()
    for k in all_keys:

        if k in ("length", "concept:name"):
            if k == "concept:name":
                names.append("Case_ID")
            else:
                names.append(k)
        else:
            if not k[0]:
                # if the first one is None, which means that we have aggregation over all events!
                name = k[1] + "|" + k[2] + "|" + k[3]
                names.append(name)
            else:
                name = k[0] + "|" + k[1] + "|" + k[2] + "|" + k[3]
                names.append(name)

            if (k[1], k[3]) in settings["one_hot_encode"]:
                selected_one_hot.add(name)

    # Given keys, go through all data and create dataframes!
    for trace_data in train_data:
        trace_df_data = []
        for k in all_keys:
            handle_data(k, trace_data, trace_df_data, settings)

        train_df_data.append(trace_df_data)

    for trace_data in test_data:
        trace_df_data = []
        for k in all_keys:
            handle_data(k, trace_data, trace_df_data, settings)

        test_df_data.append(trace_df_data)

    train_df = pd.DataFrame(train_df_data, columns=names)
    test_df = pd.DataFrame(test_df_data, columns=names)


    # save transformations to file
    transformations = []
    # Do one hot encodings, if needed
    if len(selected_one_hot) > 0:
        for selection in selected_one_hot:
            train_df[selection] = pd.Categorical(train_df[selection])
            test_df[selection] = pd.Categorical(test_df[selection])

            le = LabelEncoder()
            le.fit(train_df[selection])

            classes = le.classes_
            test_df[selection] = [x if x in set(classes) else "missing" for x in test_df[selection]]

            transformations.append(classes)

            train_df[selection] = le.transform(train_df[selection])
            test_df[selection] = le.transform(test_df[selection])

            ohe = OneHotEncoder()
            ohe.fit(train_df[selection].values.reshape(-1, 1))

            train_transformed = ohe.transform(train_df[selection].values.reshape(-1, 1)).toarray()
            test_transformed = ohe.transform(test_df[selection].values.reshape(-1, 1)).toarray()

            #print(len([selection + "_" + classes[i] for i in range(train_transformed.shape[1])]))

            dfOneHot = pd.DataFrame(train_transformed, columns=[selection + "_" + classes[i] for i in range(train_transformed.shape[1])])

            train_df = pd.concat([train_df, dfOneHot], axis=1)
            train_df.pop(selection)

            dfOneHot = pd.DataFrame(test_transformed, columns=[selection + "_" + classes[i] for i in range(train_transformed.shape[1])])
            test_df = pd.concat([test_df, dfOneHot], axis=1)
            test_df.pop(selection)

    # Save onehot transformations
    # Write transformations into file
    with open("onehot_transformations.txt", "w") as f:
        for i, selection in enumerate(selected_one_hot):
            f.write("Feature:" + selection + "\n")

            for nr, cls in enumerate(transformations[i]):
                f.write(str(nr) + ":" + str(cls) + "\n")


    train_df.to_csv("debug_train_payload.csv", index=False)
    test_df.to_csv("debug_test_payload.csv", index=False)

    for column in train_df.columns:
        if column != "concept:name":
            train_df[column] = pd.to_numeric(train_df[column], errors="ignore")
            test_df[column] = pd.to_numeric(test_df[column], errors="ignore")


    return train_df, test_df


def payload_extractor(inp_folder, log_name, settings_file):

    log = read_XES_log(log_name)

    train, test = split_log_train_test(log, 0.8)

    print("Lengths of logs train: {}, test: {}".format(len(train), len(test)))
    pex = PayloadExtractor()

    settings = settings_from_cfg(settings_file)

    ## Get first forms of data
    extracted_train_data = pex.get_overview_of_data(train, settings)
    extracted_test_data = pex.get_overview_of_data(test, settings)

    train_df, test_df = build_dataframes(extracted_train_data, extracted_test_data, settings)

    # Force all to float (except label?)

    train_df.to_csv(inp_folder + "/payload_train.csv", index=False)
    test_df.to_csv(inp_folder + "/payload_test.csv", index=False)

    return train_df, test_df




def move_payload_files(inp_folder, output_folder, split_nr):
    source = inp_folder # './baselineOutput/'
    dest1 = './' + output_folder + '/split' + str(split_nr) + "/payload/"
    files = os.listdir(source)
    for f in files:
        shutil.move(source + f, dest1)


def run_payload_extractor(log_path, settings_file, results_folder):
    for logNr in range(5):
        logPath = log_path.format(logNr + 1)
        folder_name = "./payloadOutput/"
        if not os.path.exists(folder_name):
            os.makedirs(folder_name)

        payload_extractor(folder_name, logPath, settings_file)
        move_payload_files(folder_name, results_folder, logNr + 1)



def payload_extractor_trial(log_name, settings_file):
    log = read_XES_log(log_name)

    train, test = split_log_train_test(log, 0.8)

    print("Lengths of logs train: {}, test: {}".format(len(train), len(test)))
    pex = PayloadExtractor()

    settings = settings_from_cfg(settings_file)

    ## Get first forms of data
    extracted_train_data = pex.get_overview_of_data(train, settings)
    extracted_test_data = pex.get_overview_of_data(test, settings)

    train_df, test_df = build_dataframes(extracted_train_data, extracted_test_data, settings)

    return train_df, test_df

if __name__ == "__main__":

    log = "logs/sepsis_tagged_er.xes"
    settings = "sepsis_settings.cfg"
    train_df, test_df = payload_extractor_trial(log_name = log, settings_file = settings)

    #print(test_df)