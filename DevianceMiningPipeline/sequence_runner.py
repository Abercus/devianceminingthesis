"""
Python wrapper to extract sequence encodings from logs.
Requires GoSwift.jar in the same folder as it runs it.
Might need to change VMOptions dependent on the version of Java the machine is running on

"""
import subprocess
import time



import shutil
import os


JAR_NAME = "GoSwift.jar"  # Jar file to run
OUTPUT_FOLDER = "outputlogs/"  # Where to put output files
INPUT_FOLDER = "logs/"  # Where input logs are located

## This is needed for one java version.. One for Java 8 and other for later
#VMoptions = " --add-modules java.xml.bind"
VMOptions = ""
## All parameters to run the program with.


def create_output_filename(input_log, name):
    """
    Create output json file name corresponding to the trial parameters
    :param input_log: input log filenae
    :param name: name of the trial
    :return:
    """
    prefix = input_log
    if (input_log.endswith(".xes")):
        prefix = prefix[:prefix.find(".xes")]

    filename = prefix + "_" + name + ".json"

    return filename


def create_call_params(paramString, inputFile=None, outputFile=None):
    params = paramString.split()

    if outputFile:
        params.append("--outputFile")
        params.append(OUTPUT_FOLDER + outputFile)
    if inputFile:
        params.append("--logFile")
        params.append(INPUT_FOLDER + inputFile[0])
        if inputFile[1]:
            params.append("--requiresLabelling")

    return params


def call_params(paramString, inputFile, outputFile):
    """
    Function to call java subprocess
    TODO: Send sigkill when host process (this one dies) to also kill the subprocess calls
    :param paramString:
    :param inputFile:
    :return:
    """

    print("Started working on {}".format(inputFile[0]))
    parameters = create_call_params(paramString, inputFile, outputFile)
    FNULL = open(os.devnull, 'w')  # To write output to devnull, we dont care about it

    # No java 8
    #subprocess.call(["java", "-jar", "--add-modules", "java.xml.bind", JAR_NAME] + parameters, stdout=FNULL,
    #                stderr=open("errorlogs/error_" + outputFile, "w"))  # blocking

    # Java 8
    subprocess.call(["java", "-jar", JAR_NAME] + parameters, stdout=FNULL,
                    stderr=open("errorlogs/error_" + outputFile, "w"))  # blocking
    print("Done with {}".format(str(parameters)))


def move_files(split_nr, folder, results_folder):
    """
    Move generated encodings
    :param split_nr: number of cv split
    :param folder: folder for encoding at end location
    :param results_folder: resulting folder
    :return:
    """
    source = './output/'
    dest1 = './' + results_folder + '/split' + str(split_nr) + "/" + folder + "/"

    files = os.listdir(source)

    ## Moves all files in the folder to detination
    for f in files:
        shutil.move(source+f, dest1)


def run_sequences(log_path, results_folder, sequence_threshold=5):
    """
    Runs GoSwift.jar with 4 different sets of parameters, to create sequential encodings.
    :param log_path:
    :param results_folder:
    :param sequence_threshold:
    :return:
    """

    ## Input parameters to GoSwift.jar
    paramStrings = [
        ("--coverageThreshold {} ".format(sequence_threshold) + "--featureType Sequence --minimumSupport 0.1 --patternType MR --encodingType Frequency", "SequenceMR", "mr"),
        ("--coverageThreshold {} ".format(sequence_threshold) + "--featureType Sequence --minimumSupport 0.1 --patternType MRA --encodingType Frequency", "SequenceMRA", "mra"),
        ("--coverageThreshold {} ".format(sequence_threshold) + "--featureType Sequence --minimumSupport 0.1 --patternType TR --encodingType Frequency", "SequenceTR", "tr"),
        ("--coverageThreshold {} ".format(sequence_threshold) + "--featureType Sequence --minimumSupport 0.1 --patternType TRA --encodingType Frequency", "SequenceTRA", "tra"),
    ]

    for paramString, techName, folder in paramStrings:
        print("Working on {}".format(techName))
        for splitNr in range(5):
            
            folder_name = "./output/"
            if not os.path.exists(folder_name):
                os.makedirs(folder_name)

            print("Working on split {}".format(splitNr+1))
            inputFile = (log_path.format(splitNr+1), False)
            outputFilename = create_output_filename(inputFile[0], techName)
            tic = time.time()
            call_params(paramString, inputFile, outputFilename)
            toc = time.time()
            print("Time taken {0:.3f} seconds".format(toc - tic))

            move_files(splitNr + 1, folder, results_folder)
