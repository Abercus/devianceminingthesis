from opyenxes.data_in.XUniversalParser import XUniversalParser
from opyenxes.data_out.XesXmlSerializer import XesXmlSerializer
from opyenxes.model.XAttributeDiscrete import XAttributeDiscrete
from opyenxes.model.XAttributeLiteral import XAttributeLiteral
from opyenxes.factory.XFactory import XFactory
from random import shuffle

 

def merge_and_label(normLogs, devLogs):
    
    assert(len(normLogs)>0 and len(devLogs) > 0)

    merged_log = XFactory.create_log(normLogs[0].get_attributes().clone())
    

    for elem in normLogs[0].get_extensions():
        merged_log.get_extensions().add(elem)

    merged_log.__classifiers = normLogs[0].get_classifiers().copy()
    merged_log.__globalTraceAttributes = normLogs[0].get_global_trace_attributes().copy()
    merged_log.__globalEventAttributes = normLogs[0].get_global_event_attributes().copy()
    
    merged_log.get_global_trace_attributes().append(XAttributeLiteral("Label", "0"))

    for log in normLogs:
        for trace in log:
            trace.get_attributes()["Label"] = XAttributeLiteral("Label", "0")
            merged_log.append(trace)


    for log in devLogs:
        for trace in log:
            trace.get_attributes()["Label"] = XAttributeLiteral("Label", "1")
            merged_log.append(trace)


    return merged_log





if __name__ == "__main__":
    
    normLogs = []
    devLogs = []
    
    normPathTemp = "out_xraynorm{}.xes"
    devPathTemp = "out_xraydev{}.xes"
    normLogCount = 7
    devLogCount = 1

    for i in range(1,normLogCount+1):
        logPath = normPathTemp.format(i)
        with open(logPath) as log_file:
            normLogs.append(XUniversalParser().parse(log_file)[0])

    for i in range(1, devLogCount+1):
        logPath = devPathTemp.format(i)
        with open(logPath) as log_file:
            devLogs.append(XUniversalParser().parse(log_file)[0])



    merged_log = merge_and_label(normLogs, devLogs)
    
    shuffle(merged_log)

    with open("merged_xray.xes", "w") as f:
        XesXmlSerializer().serialize(merged_log, f)



