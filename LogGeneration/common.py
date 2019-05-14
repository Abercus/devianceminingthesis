"""
Common methods for creation of xes log and traces
"""

import xes
import datetime
from random import shuffle

class LogGenerator:

    def __init__(self, config, traces=None):
        self.config = config
        if not traces:
            self.traces = []
        else:
            self.traces = traces

    def add_trace(self, trace):
        self.traces.append(trace)

    def convert_to_xes(self):
        log = create_xes_log(self.config)

        for trace in self.traces:
            xes_trace = create_xes_trace(trace_dict=trace, config=self.config)
            log.add_trace(xes_trace)

        if "shuffle" in self.config and self.config["shuffle"]:
            shuffle(log.traces)

        return log

    def create_from_event_lists(self, event_lists):
        self.traces = []


        ts = TimestampGenerator()
        for nr, event_list in enumerate(event_lists):
            trace_name = "trace_" + str(nr)
            deviant = event_list[1]
            events = event_list[0]
            # convert event_list to trace
            transformed_events = []
            for event in events:
                timestamp = ts.get_next_timestamp().astimezone().isoformat()
                name = event
                resource = "res0"
                lifecycle = "complete"
                transformed_events.append({
                    "timestamp" : timestamp,
                    "concept:name" : name,
                    "org:resource" : resource,
                    "lifecycle:transition" : lifecycle
                })
            trace = {
                "name" : trace_name,
                "deviant" : deviant,
                "events": transformed_events
            }

            self.add_trace(trace)


class TimestampGenerator:

    def __init__(self):
        self.currentTime = datetime.datetime.now(datetime.timezone.utc)


    def get_next_timestamp(self):
        return_timestamp = self.currentTime
        self.currentTime = self.currentTime + datetime.timedelta(minutes=10)
        return return_timestamp


"""
Trace consists of

a dictionary with arguments with one being a list of dictionaries

Timestamp has to be in iso format
{
"name" : name,
"deviant" : Boolean
"events" : [
    {"timestamp" : timestamp ..,
    "name" : name,
    "resource" : resource,
    "lifecycle" : lifecycle},
    {},
    {}...
    ]

}

Outputs xes.Trace class object consisting of such structure
"""
def create_xes_trace(trace_dict, config):
    xes_trace = xes.Trace()

    ## Add trace metadata
    trace_name = trace_dict["name"]
    deviant = trace_dict["deviant"]

    xes_trace.add_attribute(xes.Attribute(type="string", key="concept:name", value=trace_name))
    xes_trace.add_attribute(xes.Attribute(type=config["type"], key=config["label"],
                                          value=config["deviant"] if deviant else config["nondeviant"]))

    ## Add events
    events = trace_dict["events"]
    for event in events:
        event_timestamp = event["timestamp"]
        event_name = event["concept:name"]
        event_resource = event["org:resource"]
        event_lifecycle = event["lifecycle:transition"]


        xes_event = xes.Event()
        xes_event.attributes = [
            xes.Attribute(type="date",   key="time:timestamp", value=event_timestamp),
            xes.Attribute(type="string", key="concept:name", value=event_name),
            xes.Attribute(type="string", key="org:resource", value=event_resource),
            xes.Attribute(type="string", key="lifecycle:transition", value=event_lifecycle)
        ]
        xes_trace.add_event(xes_event)
    return xes_trace


"""
Function to create xes log with following global attributes:
Label and Date
"""
def create_xes_log(config):
    xes_log = xes.Log()
    # Add log attributes
    xes_log.add_global_event_attribute(xes.Attribute(type="date", key="time:timestamp", value=datetime.datetime.now().astimezone().isoformat()))
    #xes_log.add_global_trace_attributes(xes.Attribute(type="int", key="Label", value="0")) # For deviant, nondeviant case.
    xes_log.add_global_trace_attributes(xes.Attribute(type=config["type"], key=config["label"], value=config["nondeviant"])) # For deviant, nondeviant case.

    return xes_log


def create_log_from_trace_dicts(traces):
    log = create_xes_log()
    for trace in traces:
        log.add_trace(create_xes_trace(trace))

    return log


def generate_activity_names(count):
    return ["activity_" + str(i + 1) for i in range(count)]


def generate_trace_names(count):
    #return ["trace_" + str(i + 1) for i in range(count)]
    return [str(i + 1) for i in range(count)]


def generate_event_timestamps(count):
    timestamps = []
    startTime = datetime.datetime.now(datetime.timezone.utc)
    timestamps.append(startTime)
    for _ in range(count-1):
        currentTime = startTime + datetime.timedelta(minutes=10)
        timestamps.append(currentTime)

    return timestamps