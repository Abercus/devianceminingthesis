

def activity(name):
    return "activity {}\n".format(name)

def existence(name):
    return "Existence[{}] | |\n".format(name)

def response(name1, name2):
    return "Response[{}, {}] | |\n".format(name1, name2)

def not_response(name1, name2):
    return "NotResponse[{}, {}] | |\n".format(name1, name2)

def not_chain_response(name1, name2):
    return "NotChainResponse[{}, {}] | |\n".format(name1, name2)

def chain_response(name1, name2):
    return "ChainResponse[{}, {}] | |\n".format(name1, name2)

def absence(name):
    return "Absence[{}] | |\n".format(name)

def absence_c(name, c):
    return "Absence[{},{}] | |\n".format(name, c)

def generate_random_activities(n=100):
    activity_names = []
    for i in range(1, n+1):
        activity_names.append("random_{}".format(i))

    return activity_names


def gen_deviant_1():
    
    t_names = ["seq1", "seq2", "seq3", "seq4"]
    r_names = generate_random_activities(100)

    activity_names = t_names + r_names
    with open("dev1.decl", "w") as f:
        for name in activity_names:
            f.write(activity(name))
        f.write("\n")
        # write constraints

        f.write(existence("seq1"))
        f.write(chain_response("seq1", "seq2"))
        f.write(chain_response("seq2", "seq3"))
        f.write(chain_response("seq3", "seq4"))
        
        # write absences
        for name in activity_names:
            f.write(absence_c(name, 4))


def gen_deviant_2():
    
    t_names = ["seq1", "seq2", "seq3", "seq4"]
    r_names = generate_random_activities(100)

    activity_names = t_names + r_names
    with open("dev2.decl", "w") as f:
        for name in activity_names:
            f.write(activity(name))
        f.write("\n")
        # write constraints

        f.write(existence("seq4"))
        f.write(chain_response("seq4", "seq3"))
        f.write(chain_response("seq3", "seq2"))
        f.write(chain_response("seq2", "seq1"))
        
        # write absences
        for name in activity_names:
            f.write(absence_c(name, 4))




def gen_norm_1():
    t_names = ["seq1", "seq2", "seq3", "seq4"]
    r_names = generate_random_activities(100)
    
    activity_names = t_names + r_names
    with open("norm1.decl", "w") as f:
        for name in activity_names:
            f.write(activity(name))
        f.write("\n")
        # write constraitns
        f.write(existence("seq1"))
        f.write(existence("seq2"))
        f.write(existence("seq3"))
        f.write(existence("seq4"))
        f.write(not_chain_response("seq2", "seq3"))
        f.write(not_chain_response("seq3", "seq2"))
        # write absences
        for name in activity_names:
            f.write(absence_c(name, 4))


def gen_norm_2():
    t_names = ["seq1", "seq2", "seq3", "seq4"]
    r_names = generate_random_activities(100)
    
    activity_names = t_names + r_names
    with open("norm2.decl", "w") as f:
        for name in activity_names:
            f.write(activity(name))
        f.write("\n")
        # write constraints
        f.write(absence("seq1"))
        f.write(absence("seq2"))
        f.write(existence("seq3"))
        f.write(existence("seq4"))
        
        # write absences
        for name in activity_names:
            f.write(absence_c(name, 3))

def gen_norm_3():
    t_names = ["seq1", "seq2", "seq3", "seq4"]
    r_names = generate_random_activities(100)
    
    activity_names = t_names + r_names
    with open("norm3.decl", "w") as f:
        for name in activity_names:
            f.write(activity(name))
        f.write("\n")
        # write constraints
        f.write(not_chain_response("seq1", "seq2"))


        # write absences
        for name in activity_names:
            f.write(absence_c(name, 4))


def gen_norm_4():
    t_names = ["seq1", "seq2", "seq3", "seq4"]
    r_names = generate_random_activities(100)
    
    activity_names = t_names + r_names
    with open("norm4.decl", "w") as f:
        for name in activity_names:
            f.write(activity(name))
        f.write("\n")
        # write constraitns
        f.write(chain_response("seq1", "seq3"))
        f.write(chain_response("seq3", "seq2"))
        f.write(chain_response("seq2", "seq4"))
        f.write(existence("seq1"))

        # write absences
        for name in activity_names:
            f.write(absence_c(name, 4))




import sys

if __name__ == "__main__":
    arg = sys.argv[1]
   
    # Given decl model, get all activities, add absences..
    print("Opening {}".format(arg))
    max_each = 3

    activities = []
    with open(arg, "r") as f:
        # Find activities
        for line in f:
            if line.startswith("activity"):
                activity_name = line[line.find(" ")+1:].strip()
                activities.append(activity_name)

    with open(arg, "a") as f:
        for activity in activities:
            f.write(absence_c(activity, max_each))
    print(activities)
