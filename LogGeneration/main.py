"""
Main program for synthetic log generator for BPM area.

Generates .xes files

@author Joonas Puura

"""
from scenarios.simple import SingleActivityScenario, ActivitySetScenario

LOGS_FOLDER = "logs/"


def parse_input():
    pass


if __name__ == "__main__":
    #parse_input()
    config_prev = {
        "label" : "Label",
        "deviant" : str(1),
        "nondeviant" : str(0),
        "type" : "int"
    }


    config = {
        "label" : "class",
        "deviant" : "true",
        "nondeviant" : "false",
        "type" : "string",
        "shuffle" : True
    }


    config_2 = {
        "label" : "Label",
        "deviant" : str(1),
        "nondeviant" : str(0),
        "type" : "string",
        "shuffle" : True
    }



    #open(LOGS_FOLDER + "minimum_test.xes", "w").write(str(SingleActivityScenario.minimum_test(config=config)))
    #open(LOGS_FOLDER + "class_single_extra_1.xes", "w").write(str(SingleActivityScenario.single_activity_extra_1(config=config)))
    #open(LOGS_FOLDER + "class_single_missing_1.xes", "w").write(str(SingleActivityScenario.single_activity_missing_1(config=config)))
    #open(LOGS_FOLDER + "class_activity_set_co_occur.xes", "w").write(str(ActivitySetScenario.activity_set_co_occur(config=config)))
    #open(LOGS_FOLDER + "init_test.xes", "w").write(str(SingleActivityScenario.init_test_scenario(config=config_2)))