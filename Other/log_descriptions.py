"""
Counts traces, events, etc. to describe datasets.

"""
import opyenxes

def read_XES_log(path):
    tic = time()

    print("Parsing log")
    with open(path) as log_file:
        log = XUniversalParser().parse(log_file)[0]  # take first log from file

    toc = time()

    print("Log parsed, took {} seconds..".format(toc - tic))

    return log

def count_cases(log):

    total = len(log)
    deviant = 0
    normal = 0
    for trace in log:
        attribs = trace.get_attributes()
        if str(attribs["Label"]) == str(1):
            deviant += 1
        elif str(attribs["Label"]) == str(0):
            normal += 1

    return normal, deviant, total


def count_total_events(log):
    deviant_events = 0
    normal_events = 0
    for trace in log:
        attribs = trace.get_attributes()
        events_length = len(trace)
        if str(attribs["Label"]) == str(1):
            deviant_events += events_length
        elif str(attribs["Label"]) == str(0):
            normal_events += events_length

    return normal_events, deviant_events


def calc_avg(event_count, trace_count):
    return int(round(event_count / trace_count))


def describe_xray():

    print("XRAY DESCRIPTION")
    log = read_XES_log("logs/merged_xray.xes")
    # count total cases
    # count normal cases
    # count deviant cases
    normal, deviant, total = count_cases(log)
    print("Norm, Deviant, Total", normal, deviant, total)
    # count avg length deviant
    # count avg length norm
    normal_events, deviant_events = count_total_events(log)
    print("Avg norm {}, dev {}".format(calc_avg(normal_events, normal), calc_avg(deviant_events, deviant)))
    print("Avg total {}".format(calc_avg(normal_events+deviant_events, total)))
    # deviance criterion

def describe_sequence_mr():

    print("SYNTH MR/TR DESCRIPTION")
    log = read_XES_log("logs/synth_tr_tagged.xes")
    # count total cases
    # count normal cases
    # count deviant cases
    normal, deviant, total = count_cases(log)
    print("Norm, Deviant, Total", normal, deviant, total)
    # count avg length deviant
    # count avg length norm
    normal_events, deviant_events = count_total_events(log)
    print("Avg norm {}, dev {}".format(calc_avg(normal_events, normal), calc_avg(deviant_events, deviant)))
    print("Avg total {}".format(calc_avg(normal_events+deviant_events, total)))

    # deviance criterion


def describe_sequence_mra():

    print("SYNTH MRA/TRA DESCRIPTION")
    log = read_XES_log("logs/synth_mra_tagged.xes")
    # count total cases
    # count normal cases
    # count deviant cases
    normal, deviant, total = count_cases(log)
    print("Norm, Deviant, Total", normal, deviant, total)
    # count avg length deviant
    # count avg length norm
    normal_events, deviant_events = count_total_events(log)
    print("Avg norm {}, dev {}".format(calc_avg(normal_events, normal), calc_avg(deviant_events, deviant)))
    print("Avg total {}".format(calc_avg(normal_events+deviant_events, total)))

    # deviance criterion


def describe_bpi2011_cc():
    print("BPI2011 CC DESCRIPTION")
    log = read_XES_log("logs/EnglishBPIChallenge2011_tagged_cc.xes")
    # count total cases
    # count normal cases
    # count deviant cases
    normal, deviant, total = count_cases(log)
    print("Norm, Deviant, Total", normal, deviant, total)
    # count avg length deviant
    # count avg length norm
    normal_events, deviant_events = count_total_events(log)
    print("Avg norm {}, dev {}".format(calc_avg(normal_events, normal), calc_avg(deviant_events, deviant)))
    print("Avg total {}".format(calc_avg(normal_events+deviant_events, total)))

    # deviance criterion


def describe_bpi2011_m13():
    print("BPI2011 M13 DESCRIPTION")
    log = read_XES_log("logs/EnglishBPIChallenge2011_tagged_M13.xes")
    # count total cases
    # count normal cases
    # count deviant cases
    normal, deviant, total = count_cases(log)
    print("Norm, Deviant, Total", normal, deviant, total)
    # count avg length deviant
    # count avg length norm
    normal_events, deviant_events = count_total_events(log)
    print("Avg norm {}, dev {}".format(calc_avg(normal_events, normal), calc_avg(deviant_events, deviant)))
    print("Avg total {}".format(calc_avg(normal_events+deviant_events, total)))

    # deviance criterion


def describe_bpi2011_t101():
    print("BPI2011 t101 DESCRIPTION")
    log = read_XES_log("logs/EnglishBPIChallenge2011_tagged_t101.xes")
    # count total cases
    # count normal cases
    # count deviant cases
    normal, deviant, total = count_cases(log)
    print("Norm, Deviant, Total", normal, deviant, total)
    # count avg length deviant
    # count avg length norm
    normal_events, deviant_events = count_total_events(log)
    print("Avg norm {}, dev {}".format(calc_avg(normal_events, normal), calc_avg(deviant_events, deviant)))
    print("Avg total {}".format(calc_avg(normal_events+deviant_events, total)))

    # deviance criterion

def describe_bpi2011_m16():
    print("BPI2011 M16 DESCRIPTION")
    log = read_XES_log("logs/EnglishBPIChallenge2011_tagged_m16.xes")
    # count total cases
    # count normal cases
    # count deviant cases
    normal, deviant, total = count_cases(log)
    print("Norm, Deviant, Total", normal, deviant, total)
    # count avg length deviant
    # count avg length norm
    normal_events, deviant_events = count_total_events(log)
    print("Avg norm {}, dev {}".format(calc_avg(normal_events, normal), calc_avg(deviant_events, deviant)))
    # deviance criterion
    print("Avg total {}".format(calc_avg(normal_events+deviant_events, total)))


def describe_sepsis_er():
    print("SEPSIS ER DESCRIPTION")
    log = read_XES_log("logs/sepsis_tagged_er.xes")
    # count total cases
    # count normal cases
    # count deviant cases
    normal, deviant, total = count_cases(log)
    print("Norm, Deviant, Total", normal, deviant, total)
    # count avg length deviant
    # count avg length norm
    normal_events, deviant_events = count_total_events(log)
    print("Avg norm {}, dev {}".format(calc_avg(normal_events, normal), calc_avg(deviant_events, deviant)))
    print("Avg total {}".format(calc_avg(normal_events+deviant_events, total)))

    # deviance criterion

def describe_sepsis_sequence():
    print("SEPSIS SEQUENCE DESCRIPTION")
    log = read_XES_log("logs/sepsis_sequence_tagged.xes")
    # count total cases
    # count normal cases
    # count deviant cases
    normal, deviant, total = count_cases(log)
    print("Norm, Deviant, Total", normal, deviant, total)
    # count avg length deviant
    # count avg length norm
    normal_events, deviant_events = count_total_events(log)
    print("Avg norm {}, dev {}".format(calc_avg(normal_events, normal), calc_avg(deviant_events, deviant)))
    # deviance criterion
    print("Avg total {}".format(calc_avg(normal_events+deviant_events, total)))


def describe_sepsis_declare():
    print("SEPSIS DECLARE DESCRIPTION")
    log = read_XES_log("logs/sepsis_constraint_tagged.xes")
    # count total cases
    # count normal cases
    # count deviant cases
    normal, deviant, total = count_cases(log)
    print("Norm, Deviant, Total", normal, deviant, total)
    # count avg length deviant
    # count avg length norm
    normal_events, deviant_events = count_total_events(log)
    print("Avg norm {}, dev {}".format(calc_avg(normal_events, normal), calc_avg(deviant_events, deviant)))
    print("Avg total {}".format(calc_avg(normal_events+deviant_events, total)))

    # deviance criterion


if __name__ == "__main__":
    describe_xray()
    describe_sequence_mr()
    describe_sequence_mra()
    describe_bpi2011_cc()
    describe_bpi2011_m13()
    describe_bpi2011_m16()
    describe_bpi2011_t101()
    describe_sepsis_declare()
    describe_sepsis_er()
    describe_sepsis_sequence()

