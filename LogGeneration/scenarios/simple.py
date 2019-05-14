"""
This file describes testing scenarios
"""
from common import create_xes_log, generate_activity_names, generate_trace_names, TimestampGenerator, LogGenerator
from random import shuffle, sample, choice, randint

"""
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
"""

class SingleActivityScenario:


    @staticmethod
    def init_test_scenario(config):
        log = LogGenerator(config)
        activity_count = 20
        nr_non_deviant = 100
        nr_deviant = 100

        activity_names = generate_activity_names(activity_count)
        trace_names = generate_trace_names(nr_deviant + nr_non_deviant)

        # choose deviant init activity
        deviant_init_activity = activity_names[0]

        for i in range(nr_non_deviant + nr_deviant):
            trace = {
                "deviant" : i >= nr_non_deviant,
                "name" : trace_names[i],
            }

            timestamp_generator = TimestampGenerator()
            events = []

            for j in range(activity_count):
                event = {
                    "org:resource": "resource1",
                    "lifecycle:transition": "COMPLETE",
                    "concept:name": activity_names[j]
                }
                if activity_names[j] == deviant_init_activity:
                    continue
                else:
                    events.append(event)


            shuffle(events) #shuffle first

            # add our last activity..
            event = {
                "org:resource": "resource1",
                "lifecycle:transition": "COMPLETE",
                "concept:name": deviant_init_activity
            }

            if trace["deviant"]:
                # if deviant, then put to first...
                events.insert(0, event)
            else:
                # put to random place, which not first
                events.insert(randint(1, len(events)), event)

            # add timestamps
            for event in events:
                event["timestamp"] = timestamp_generator.get_next_timestamp().astimezone().isoformat()


            trace["events"] = events
            log.add_trace(trace)

        return log.convert_to_xes()


    @staticmethod
    def minimum_test(config):
        log = LogGenerator(config)
        activity_count = 2
        nr_non_deviant = 10
        nr_deviant = 10

        activity_names = generate_activity_names(activity_count)
        trace_names = generate_trace_names(nr_deviant + nr_non_deviant)
        added_in_deviant = activity_names[0]  # remove event with activity name

        for i in range(nr_non_deviant + nr_deviant):  #
            trace = {}
            trace["deviant"] = i >= nr_non_deviant
            trace["name"] = trace_names[i]
            events = []
            timestamp_generator = TimestampGenerator()

            for j in range(activity_count):
                # dont add activity in deviant cases
                if trace["deviant"] and activity_names[j] == added_in_deviant:
                    continue
                event = {
                    "org:resource": "resource1",
                    "lifecycle:transition": "COMPLETE",
                    "concept:name": activity_names[j]
                }
                events.append(event)

            # shuffle events, in order not be able to learn sequences of larger size than 1
            shuffle(events)
            # add timestamps
            for event in events:
                event["timestamp"] = timestamp_generator.get_next_timestamp().astimezone().isoformat()

            trace["events"] = events
            log.add_trace(trace)

        return log.convert_to_xes()


    @staticmethod
    def single_activity_missing_1(config):
        """
        One activity missing causes the traces to be deviant
        :return:
        """
        log = LogGenerator(config)

        activity_count = 15
        nr_non_deviant = 100
        nr_deviant = 100

        activity_names = generate_activity_names(activity_count)
        trace_names = generate_trace_names(nr_deviant + nr_non_deviant)

        added_in_deviant = activity_names[5]  # remove event with activity name

        for i in range(nr_non_deviant + nr_deviant):  #
            trace = {}
            trace["deviant"] = i >= nr_non_deviant
            trace["name"] = trace_names[i]
            events = []
            timestamp_generator = TimestampGenerator()

            for j in range(activity_count):
                # dont add activity in deviant cases
                if trace["deviant"] and activity_names[j] == added_in_deviant:
                    continue
                event = {
                    "org:resource": "resource1",
                    "lifecycle:transition": "COMPLETE",
                    "concept:name": activity_names[j]
                }
                events.append(event)

            # shuffle events, in order not be able to learn sequences of larger size than 1
            shuffle(events)
            # add timestamps
            for event in events:
                event["timestamp"] = timestamp_generator.get_next_timestamp().astimezone().isoformat()

            trace["events"] = events
            log.add_trace(trace)

        return log.convert_to_xes()


    @staticmethod
    def single_activity_extra_1(config):
        """
        Addition of one activity causes the trace to be deviant
        :return:
        """
        log = LogGenerator(config)

        activity_count = 15
        nr_non_deviant = 100
        nr_deviant = 100

        activity_names = generate_activity_names(activity_count)
        trace_names = generate_trace_names(nr_deviant + nr_non_deviant)

        added_in_deviant = activity_names[5] # remove event with activity name

        for i in range(nr_non_deviant + nr_deviant): #
            trace = {}
            trace["deviant"] = i >= nr_non_deviant
            trace["name"] = trace_names[i]
            events = []
            timestamp_generator = TimestampGenerator()

            for j in range(activity_count):
                # dont add activity in non-deviant cases
                if not trace["deviant"] and activity_names[j] == added_in_deviant:
                    continue
                event = {
                    "org:resource" : "resource1",
                    "lifecycle:transition" : "COMPLETE",
                    "concept:name" : activity_names[j]
                }
                events.append(event)

            # shuffle events, in order not be able to learn sequences of larger size than 1
            shuffle(events)
            # add timestamps
            for event in events:
                event["timestamp"] = timestamp_generator.get_next_timestamp().astimezone().isoformat()

            trace["events"] = events
            log.add_trace(trace)

        return log.convert_to_xes()




class ActivitySetScenario:
    @staticmethod
    def activity_set_co_occur(config):
        """
        A set of activities occurring together causes the trace to be deviant.
        Ex if events A,B,C and occurring together in the trace, then the trace is deviant.
        If only B,C or A,B .. then the trace is not deviant.
        :return:
        """
        log = LogGenerator(config)

        activity_count = 10
        nr_non_deviant = 100
        nr_deviant = 50

        activity_names = generate_activity_names(activity_count)
        trace_names = generate_trace_names(nr_deviant + nr_non_deviant)

        deviant_set = (activity_names[4], activity_names[5], activity_names[6]) # remove event with activity name

        for i in range(nr_non_deviant + nr_deviant): #
            trace = {
                "deviant" : i >= nr_non_deviant,
                "name" : trace_names[i]
                 }

            events = []
            timestamp_generator = TimestampGenerator()

            to_remove_set = None
            if not trace["deviant"]:
                # Remove randomly 1 to 3 activities inside deviant set from trace
                remove_count = randint(1,3)
                to_remove_set = sample(deviant_set, remove_count)
            else:
                # remove all
                to_remove_set = set() # remove nothing if deviant


            for j in range(activity_count):
                # dont add activity
                if activity_names[j] in to_remove_set:
                    continue
                event = {
                    "org:resource" : "resource1",
                    "lifecycle:transition" : "COMPLETE",
                    "concept:name" : activity_names[j]
                }
                events.append(event)

            # shuffle events, in order not be able to learn sequences of larger size than 1
            shuffle(events)
            # add timestamps
            for event in events:
                event["timestamp"] = timestamp_generator.get_next_timestamp().astimezone().isoformat()

            trace["events"] = events
            log.add_trace(trace)

        return log.convert_to_xes()



class ImbalancedDataScenario:
    @staticmethod
    def imbalanced_activity_set_finding_1():
        return None

class SequenceScenario:
    @staticmethod
    def sequence_extra_1():
        return None